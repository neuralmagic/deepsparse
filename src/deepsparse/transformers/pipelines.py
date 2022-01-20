# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adaptation of transformers.pipelines and onnx_transformers.pipelines

adapted from:
https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/base.py
https://github.com/patil-suraj/onnx_transformers/blob/master/onnx_transformers/pipelines.py

"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.data import (
    SquadExample,
    SquadFeatures,
    squad_convert_examples_to_features,
)
from transformers.file_utils import ExplicitEnum
from transformers.models.auto import AutoConfig, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from transformers.utils import logging

from deepsparse import Engine, compile_model, cpu
from deepsparse.transformers.helpers import (
    fix_numpy_types,
    get_onnx_path_and_configs,
    overwrite_transformer_onnx_model_inputs,
)
from deepsparse.transformers.loaders import get_batch_loader


try:
    import onnxruntime

    ort_import_error = None
except Exception as ort_import_err:
    onnxruntime = None
    ort_import_error = ort_import_err

__all__ = [
    "ArgumentHandler",
    "Pipeline",
    "TextClassificationPipeline",
    "TokenClassificationPipeline",
    "QuestionAnsweringPipeline",
    "pipeline",
    "overwrite_transformer_onnx_model_inputs",
    "SUPPORTED_ENGINES",
    "SUPPORTED_TASKS",
]

logger = logging.get_logger(__name__) if logging else None


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each Pipeline.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultArgumentHandler(ArgumentHandler):
    """
    Default argument parser handling parameters for each Pipeline`.
    """

    @staticmethod
    def handle_kwargs(kwargs: Dict) -> List:
        """
        :param kwargs: key word arguments for a pipeline
        :return: list of the processed key word arguments
        """
        if len(kwargs) == 1:
            output = list(kwargs.values())
        else:
            output = list(chain(kwargs.values()))

        return DefaultArgumentHandler.handle_args(output)

    @staticmethod
    def handle_args(args: Sequence[Any]) -> List[str]:
        """
        :param args: sequence of arguments to a pipeline
        :return: list of formatted, processed arguments
        """

        # Only one argument, let's do case by case
        if len(args) == 1:
            if isinstance(args[0], str):
                return [args[0]]
            elif not isinstance(args[0], list):
                return list(args)
            else:
                return args[0]

        # Multiple arguments (x1, x2, ...)
        elif len(args) > 1:
            if all([isinstance(arg, str) for arg in args]):
                return list(args)

            # If not instance of list, then it should be an instance of iterable
            elif isinstance(args, Iterable):
                return list(chain.from_iterable(chain(args)))
            else:
                raise ValueError(
                    f"Invalid input type {type(args)}. Pipeline supports "
                    "Union[str, Iterable[str]]"
                )
        else:
            return []

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0 and len(args) > 0:
            raise ValueError("Pipeline cannot handle mixed args and kwargs")

        if len(kwargs) > 0:
            return DefaultArgumentHandler.handle_kwargs(kwargs)
        else:
            return DefaultArgumentHandler.handle_args(args)


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()


class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit.
    Refer to this class for methods shared across different pipelines.
    This base Pipeline class provides support for multiple inference engine backends.

    Base class implementing pipelined operations.
    Pipeline workflow is defined as a sequence of the following operations:

        Input -> Tokenization -> Model Inference ->
        Post-Processing (task dependent) -> Output

    Pipeline supports running with the DeepSparse engine or onnxruntime.

    :param model: loaded inference engine to run the model with, can be a
        deepsparse Engine or onnxruntime InferenceSession
    :param tokenizer: tokenizer to be used for preprocessing
    :param config: transformers model config for this model
    :param engine_type: name of inference engine that is used. Options are
        deepsparse and onnxruntime
    :param max_length: maximum sequence length to set for model inputs by default.
        default value is 128
    :param input_names: list of input names to the neural network
    :param args_parser: Reference to the object in charge of parsing supplied
        pipeline parameters. A default is provided if None
    :param binary_output: if True, stores outputs as pickled binaries to avoid
        storing large amount of textual data. Default is False
    """

    default_input_names = None

    def __init__(
        self,
        model: Union[Engine, "onnxruntime.InferenceSession"],
        tokenizer: PreTrainedTokenizer,
        config: PretrainedConfig,
        engine_type: str,
        max_length: int = 128,
        input_names: Optional[List[str]] = None,
        args_parser: ArgumentHandler = None,
        binary_output: bool = False,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.engine_type = engine_type
        self.max_length = max_length
        self.input_names = input_names
        self.binary_output = binary_output
        self._args_parser = args_parser or DefaultArgumentHandler()
        self._framework = (
            "np" if self.engine_type in [DEEPSPARSE_ENGINE, ORT_ENGINE] else "pt"
        )

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines.
        This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines.
        This method will forward to __call__().
        """
        return self(X=X)

    def _parse_and_tokenize(
        self, *args, padding=True, add_special_tokens=True, **kwargs
    ):
        # Parse arguments
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self._framework,
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
        )

        return inputs

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs):
        if not all(name in inputs for name in self.input_names):
            raise ValueError(
                f"pipeline expected arrays with names {self.input_names}, received "
                f"inputs: {list(inputs.keys())}"
            )

        if self.engine_type == ORT_ENGINE:
            inputs = {k: v for k, v in inputs.items() if k in self.input_names}
            return self.model.run(None, inputs)
        elif self.engine_type == DEEPSPARSE_ENGINE:
            return self.model.run([inputs[name] for name in self.input_names])
        # TODO: torch
        # with self.device_placement():
        #         with torch.no_grad():
        #             inputs = self.ensure_tensor_on_device(**inputs)
        #             predictions = self.model(**inputs)[0].cpu()
        # if return_tensors:
        #     return predictions
        # else:
        #     return predictions.numpy()


class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):

        if inputs is not None and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = list(inputs)
            batch_size = len(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
            batch_size = 1
        else:
            raise ValueError("At least one input is required.")

        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(
                offset_mapping[0], tuple
            ):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
                raise ValueError(
                    "offset_mapping should have the same batch size as the input"
                )
        return inputs, offset_mapping


class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments
    (i.e. question & context) to be mapped
    to internal `transformers.SquadExample`

    QuestionAnsweringArgumentHandler manages all the possible to create a
    `transformers.SquadExample` from the command-line supplied arguments
    """

    def __call__(self, *args, **kwargs):
        # Position args, handling is sensibly the same as X and data,
        # so forwarding to avoid duplicating
        if args is not None and len(args) > 0:
            if len(args) == 1:
                kwargs["X"] = args[0]
            else:
                kwargs["X"] = list(args)

        # Generic compatibility with sklearn and Keras
        # Batched data
        if "X" in kwargs or "data" in kwargs:
            inputs = kwargs["X"] if "X" in kwargs else kwargs["data"]

            if isinstance(inputs, dict):
                inputs = [inputs]
            else:
                # Copy to avoid overriding arguments
                inputs = [i for i in inputs]

            for i, item in enumerate(inputs):
                if isinstance(item, dict):
                    if any(k not in item for k in ["question", "context"]):
                        raise KeyError(
                            "You need to provide a dictionary with keys "
                            "{question:..., context:...}"
                        )

                    inputs[i] = QuestionAnsweringPipeline.create_sample(**item)

                elif not isinstance(item, SquadExample):
                    arg_name = "X" if "X" in kwargs else "data"
                    raise ValueError(
                        f"{arg_name} argument needs to be of type "
                        "(list[SquadExample | dict], SquadExample, dict)"
                    )

        # Tabular input
        elif "question" in kwargs and "context" in kwargs:
            if isinstance(kwargs["question"], str):
                kwargs["question"] = [kwargs["question"]]

            if isinstance(kwargs["context"], str):
                kwargs["context"] = [kwargs["context"]]

            inputs = [
                QuestionAnsweringPipeline.create_sample(q, c)
                for q, c in zip(kwargs["question"], kwargs["context"])
            ]
        else:
            raise ValueError(f"Unknown arguments {kwargs}")

        if not isinstance(inputs, list):
            inputs = [inputs]

        return inputs


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using any `ModelForSequenceClassification`.

    This text classification pipeline can currently be loaded from `pipeline()`
    using the following task identifier: `"text-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on
    a text classification task.

    :param return_all_scores: set True to return all model scores. Default False
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        :param args: One or several texts (or one list of prompts) to classify
        :param args: kwargs for inner call function
        :return: A list or a list of list of dicts: Each result comes as list of dicts
            with the following keys:
            - `label` -- The label predicted.
            - `score` -- The corresponding probability.
            If ``self.return_all_scores=True``, one dictionary is returned per label
        """
        outputs = super().__call__(*args, **kwargs)

        if isinstance(outputs, list) and outputs:
            outputs = outputs[0]

        if self.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-outputs))
        else:
            scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [
                    {"label": self.config.id2label[i], "score": score.item()}
                    for i, score in enumerate(item)
                ]
                for item in scores
            ]
        else:
            return [
                {
                    "label": self.config.id2label[item.argmax()],
                    "score": item.max().item(),
                }
                for item in scores
            ]


class AggregationStrategy(ExplicitEnum):
    """
    All the valid aggregation strategies for TokenClassificationPipeline
    """

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any `ModelForTokenClassification`.

    This token classification pipeline can currently be loaded from `pipeline()`
    using the following task identifier: `"token-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on
    a token classification task.

    :param args_parser: argument parser to use default is
        TokenClassificationArgumentHandler
    :param aggregation_strategy: AggregationStrategy Enum object to determine
        the pipeline aggregation strategy. Default is AggregationStrategy.NONE
    :param ignore_labels: list of labels to ignore. Default is `["O"]`
    """

    default_input_names = "sequences"

    def __init__(
        self,
        args_parser: ArgumentHandler = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.NONE,
        ignore_labels: List[str] = False,
        **kwargs,
    ):
        super().__init__(
            args_parser=args_parser or TokenClassificationArgumentHandler(),
            **kwargs,
        )

        self.ignore_labels = ignore_labels or ["O"]

        if isinstance(aggregation_strategy, str):
            aggregation_strategy = AggregationStrategy[aggregation_strategy.upper()]

        if (
            aggregation_strategy
            in {
                AggregationStrategy.FIRST,
                AggregationStrategy.MAX,
                AggregationStrategy.AVERAGE,
            }
            and not self.tokenizer.is_fast
        ):
            raise ValueError(
                "Slow tokenizers cannot handle subwords. Please set the "
                '`aggregation_strategy` option to `"simple"` or use a fast tokenizer.'
            )

        self.aggregation_strategy = aggregation_strategy

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.


        :param inputs: One or several texts (or one list of texts) for token
            classification
        :return: A list or a list of list of :obj:`dict`: Each result comes as a list
            of dictionaries (one for each token in the corresponding input, or each
            entity if this pipeline was instantiated with an aggregation_strategy)
            with the following keys:
            - `word` -- The token/word classified.
            - `score` -- The corresponding probability for `entity`.
            - `entity` -- The entity predicted for that token/word (it is named
                `entity_group` when `aggregation_strategy` is not `"none"`.
            - `index` -- The index of the corresponding token in the sentence.
            - `start` -- index of the start of the corresponding entity in the sentence
                Only exists if the offsets are available within the tokenizer
            - `end` -- The index of the end of the corresponding entity in the sentence.
                Only exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []

        tokens = self.tokenizer(
            _inputs,
            return_tensors=self._framework,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
            padding=PaddingStrategy.MAX_LENGTH.value,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )

        if self.tokenizer.is_fast:
            offset_mapping = tokens.pop("offset_mapping")
        elif not offset_mappings:
            offset_mapping = [None] * len(_inputs)

        special_tokens_mask = tokens.pop("special_tokens_mask")

        # Forward
        _forward_pass = self._forward(tokens)
        for entities_index, current_entities in enumerate(_forward_pass[0]):
            input_ids = tokens["input_ids"][entities_index]

            scores = np.exp(current_entities) / np.exp(current_entities).sum(
                -1, keepdims=True
            )
            pre_entities = self.gather_pre_entities(
                _inputs[entities_index],
                input_ids,
                scores,
                offset_mapping[entities_index],
                special_tokens_mask[entities_index],
            )
            grouped_entities = self.aggregate(pre_entities, self.aggregation_strategy)
            # Filter anything that is in self.ignore_labels
            current_entities = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in self.ignore_labels
                and entity.get("entity_group", None) not in self.ignore_labels
            ]
            answers.append(current_entities)

        if len(answers) == 1:
            return answers[0]
        return answers

    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
    ) -> List[dict]:
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens, they should only occur
            # at the sentence boundaries since we're not encoding pairs of
            # sentences so we don't have to keep track of those.
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
                is_subword = len(word_ref) != len(word)

                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate(
        self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy
    ) -> List[dict]:
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def aggregate_word(
        self, entities: List[dict], aggregation_strategy: AggregationStrategy
    ) -> dict:
        word = self.tokenizer.convert_tokens_to_string(
            [entity["word"] for entity in entities]
        )
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def aggregate_words(
        self, entities: List[dict], aggregation_strategy: AggregationStrategy
    ) -> List[dict]:
        assert aggregation_strategy not in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }, "NONE and SIMPLE strategies are invalid"

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(
                    self.aggregate_word(word_group, aggregation_strategy)
                )
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            bi = "B"
            tag = entity_name
        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups


class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using any `ModelForQuestionAnswering`

    This question answering pipeline can currently be loaded from `pipeline()`
    using the following task identifier: `"question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on
    a question answering task.

    :param model: loaded inference engine to run the model with, can be a
        deepsparse Engine or onnxruntime InferenceSession
    :param tokenizer: tokenizer to be used for preprocessing
    :param config: transformers model config for this model
    :param engine_type: name of inference engine that is used. Options are
        deepsparse and onnxruntime
    :param input_names: list of input names to the neural network
    :param args_parser: Reference to the object in charge of parsing supplied
        pipeline parameters. A default is provided if None
    :param binary_output: if True, stores outputs as pickled binaries to avoid
        storing large amount of textual data. Default is False
    """

    default_input_names = "question,context"

    def __init__(
        self,
        model: Union[Engine, "onnxruntime.InferenceSession"],
        tokenizer: PreTrainedTokenizer,
        engine_type: str,
        input_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            engine_type=engine_type,
            args_parser=QuestionAnsweringArgumentHandler(),
            input_names=input_names,
            **kwargs,
        )

    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        :param question: single question or list of question strings
        :param context: single context or list of context strings
        :return: processed SquadExample object(s) for each question/context pair given
        """
        if isinstance(question, list):
            return [
                SquadExample(None, q, c, None, None, None)
                for q, c in zip(question, context)
            ]
        else:
            return SquadExample(None, question, context, None, None, None)

    def __call__(self, *args, **kwargs):
        """
        Answer the question(s) given as inputs by using the context(s).
        Multiple arguments can be used to pass the context, question data

        :param args: SquadExample or list of them containing the question and context
        :param X: SquadExample or list of them containing the question and context
        :param data: SquadExample or list of them containing the question and context
        :param question: single question or list of question strings
        :param context: single context or list of context strings
        :param topk: the number of answers to return. Will be chosen by
            order of likelihood)
        :param doc_stride: if the context is too long to fit with the question for the
            model, it will be split in several chunks with some overlap. This argument
            controls the size of that overlap
        :param max_answer_len: maximum length of predicted answers (e.g., only
            answers with a shorter length are considered)
        :param max_seq_len: maximum length of the total sentence (context + question)
            after tokenization. The context will be split in several chunks
            (using the doc_stride) if needed
        :param max_question_len: maximum length of the question after tokenization.
            It will be truncated if needed
        :param handle_impossible_answer: whether or not we accept impossible as an
            answer
        :param num_spans: maximum number of span to use as input from a long
            context. Default is to stride the entire context string
        :param preprocessed_inputs: if provided, preprocessing will be skipped in favor
            of these inputs. Expected format is the output of self.preprocess; a tuple
            of (examples, features_list)
        :return: dict or list of dictionaries, each containing the following keys:
            `"score"` - The probability associated to the answer
            `"start"` - The start index of the answer
            `"end"` - The end index of the answer
            `"answer"` - The answer to the question
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("handle_impossible_answer", False)
        kwargs.setdefault("preprocessed_inputs", None)  # (examples, features_list)

        if kwargs["topk"] < 1:
            raise ValueError(f"topk parameter should be >= 1 (got {kwargs['topk']})")

        if kwargs["max_answer_len"] < 1:
            raise ValueError(
                "max_answer_len parameter should be >= 1 "
                f"(got {kwargs['max_answer_len']})"
            )

        # run pre-processing if not provided
        examples, features_list = kwargs["preprocessed_inputs"] or self.preprocess(
            *args, **kwargs
        )

        # forward pass and post-processing
        all_answers = []
        for features, example in zip(features_list, examples):
            model_input_names = self.tokenizer.model_input_names + ["input_ids"]
            fw_args = {
                k: [feature.__dict__[k] for feature in features]
                for k in model_input_names
            }

            # Manage tensor allocation on correct device
            fw_args = {k: np.array(v) for (k, v) in fw_args.items()}
            start, end = self._forward(fw_args)[:2]

            # TODO: torch
            # fw_args = {k: torch.tensor(v, device=self.device)
            #   for (k, v) in fw_args.items()}
            # start, end = self.model(**fw_args)[:2]
            # start, end = start.cpu().numpy(), end.cpu().numpy()

            min_null_score = 1000000  # large and positive
            answers = []
            for (feature, start_, end_) in zip(features, start, end):
                # Ensure padded tokens & question tokens cannot belong
                undesired_tokens = (
                    np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask
                )

                # Generate mask
                undesired_tokens_mask = undesired_tokens == 0.0

                # Make sure non-context indexes cannot contribute to the softmax
                start_ = np.where(undesired_tokens_mask, -10000.0, start_)
                end_ = np.where(undesired_tokens_mask, -10000.0, end_)

                # Normalize logits and spans to retrieve the answer
                start_ = np.exp(
                    start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True))
                )
                end_ = np.exp(
                    end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True))
                )

                if kwargs["handle_impossible_answer"]:
                    min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

                # Mask CLS
                start_[0] = end_[0] = 0.0

                starts, ends, scores = self.decode(
                    start_, end_, kwargs["topk"], kwargs["max_answer_len"]
                )

                if not self.tokenizer.is_fast:
                    char_to_word = np.array(example.char_to_word_offset)
                    answers += [
                        {
                            "score": score.item(),
                            "start": np.where(
                                char_to_word == feature.token_to_orig_map[s]
                            )[0][0].item(),
                            "end": np.where(
                                char_to_word == feature.token_to_orig_map[e]
                            )[0][-1].item(),
                            "answer": " ".join(
                                example.doc_tokens[
                                    feature.token_to_orig_map[
                                        s
                                    ] : feature.token_to_orig_map[e]
                                    + 1
                                ]
                            ),
                        }
                        for s, e, score in zip(starts, ends, scores)
                    ]
                else:
                    question_first = bool(self.tokenizer.padding_side == "right")

                    # Sometimes the max probability token is in the middle of a word so:
                    # we start by finding the right word containing the token with
                    # `token_to_word` then we convert this word in a character span
                    answers += [
                        {
                            "score": score.item(),
                            "start": feature.encoding.word_to_chars(
                                feature.encoding.token_to_word(s),
                                sequence_index=1 if question_first else 0,
                            )[0],
                            "end": feature.encoding.word_to_chars(
                                feature.encoding.token_to_word(e),
                                sequence_index=1 if question_first else 0,
                            )[1],
                            "answer": example.context_text[
                                feature.encoding.word_to_chars(
                                    feature.encoding.token_to_word(s),
                                    sequence_index=1 if question_first else 0,
                                )[0] : feature.encoding.word_to_chars(
                                    feature.encoding.token_to_word(e),
                                    sequence_index=1 if question_first else 0,
                                )[
                                    1
                                ]
                            ],
                        }
                        for s, e, score in zip(starts, ends, scores)
                    ]

            if kwargs["handle_impossible_answer"]:
                answers.append(
                    {"score": min_null_score, "start": 0, "end": 0, "answer": ""}
                )

            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[
                : kwargs["topk"]
            ]
            all_answers += answers

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers

    def preprocess(self, *args, **kwargs) -> Tuple[Any, Any]:
        """
        preprocess the given QA model inputs using squad_convert_examples_to_features

        :param args: SquadExample or list of them containing the question and context
        :param X: SquadExample or list of them containing the question and context
        :param data: SquadExample or list of them containing the question and context
        :param question: single question or list of question strings
        :param context: single context or list of context strings
        :param doc_stride: if the context is too long to fit with the question for the
            model, it will be split in several chunks with some overlap. This argument
            controls the size of that overlap
        :param max_seq_len: maximum length of the total sentence (context + question)
            after tokenization. The context will be split in several chunks
            (using the doc_stride) if needed
        :param max_question_len: maximum length of the question after tokenization.
            It will be truncated if needed
        :param num_spans: maximum number of spans to use as input from a long
            context. Default is to stride the entire context string
        :return: tuple of SquadExample inputs and preprocessed features list
        """
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_seq_len", self.max_length)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("num_spans", None)

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        if not self.tokenizer.is_fast:
            features_list = [
                squad_convert_examples_to_features(
                    examples=[example],
                    tokenizer=self.tokenizer,
                    max_seq_length=kwargs["max_seq_len"],
                    doc_stride=kwargs["doc_stride"],
                    max_query_length=kwargs["max_question_len"],
                    padding_strategy=PaddingStrategy.MAX_LENGTH.value,
                    is_training=False,
                    tqdm_enabled=False,
                )
                for example in examples
            ]
        else:
            features_list = self._encode_features_fast(examples, **kwargs)

        if kwargs["num_spans"]:
            features_list = [
                features[: kwargs["num_spans"]] for features in features_list
            ]

        return examples, features_list

    def decode(
        self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int
    ) -> Tuple:
        """
        :param start: Individual start probabilities for each token
        :param end: Individual end probabilities for each token
        :param topk: Indicates how many possible answer span(s) to extract from the
            model output
        :param max_answer_len: Maximum size of the answer to extract from the model
            output
        :return: probabilities for each span to be the actual answer. Will filter out
            unwanted and impossible cases
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end]

    def span_to_answer(
        self, text: str, start: int, end: int
    ) -> Dict[str, Union[str, int]]:
        """
        When decoding from token probabilities, this method maps token indexes to
        actual word in the initial context.

        :param text: The actual context to extract the answer from
        :param start: The answer starting token index
        :param end: The answer end token index
        :return: Dictionary containing the start, end, and answer
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }

    def _encode_features_fast(self, examples: Any, **kwargs) -> List[SquadFeatures]:
        features_list = []
        for example in examples:
            # Define the side we want to truncate / pad and the text/pair sorting
            question_first = bool(self.tokenizer.padding_side == "right")

            encoded_inputs = self.tokenizer(
                text=example.question_text if question_first else example.context_text,
                text_pair=(
                    example.context_text if question_first else example.question_text
                ),
                padding=PaddingStrategy.MAX_LENGTH.value,
                truncation="only_second" if question_first else "only_first",
                max_length=kwargs["max_seq_len"],
                stride=kwargs["doc_stride"],
                return_tensors="np",
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )

            total_spans = len(encoded_inputs["input_ids"])

            # p_mask: mask with 1 for token than cannot be in the answer
            # We put 0 on the tokens from the context and 1 everywhere else
            p_mask = np.asarray(
                [
                    [
                        tok != 1 if question_first else 0
                        for tok in encoded_inputs.sequence_ids(span_id)
                    ]
                    for span_id in range(total_spans)
                ]
            )

            # keep the cls_token unmasked
            if self.tokenizer.cls_token_id is not None:
                cls_index = np.nonzero(
                    encoded_inputs["input_ids"] == self.tokenizer.cls_token_id
                )
                p_mask[cls_index] = 0

            features = []
            for span_idx in range(total_spans):
                features.append(
                    SquadFeatures(
                        input_ids=encoded_inputs["input_ids"][span_idx],
                        attention_mask=encoded_inputs["attention_mask"][span_idx],
                        token_type_ids=encoded_inputs["token_type_ids"][span_idx],
                        p_mask=p_mask[span_idx].tolist(),
                        encoding=encoded_inputs[span_idx],
                        # the following values are unused for fast tokenizers
                        cls_index=None,
                        token_to_orig_map={},
                        example_index=0,
                        unique_id=0,
                        paragraph_len=0,
                        token_is_max_context=0,
                        tokens=[],
                        start_position=0,
                        end_position=0,
                        is_impossible=False,
                        qas_id=None,
                    )
                )
            features_list.append(features)
        return features_list


@dataclass
class TaskInfo:
    """
    Information about an NLP task

    :param pipeline_constructor: reference to constructor for the given pipeline task
    :param default model name: the transformers canonical name for the default model
    :param base_stub: sparsezoo stub path for the base model for this task
    :param default_pruned_stub: sparsezoo stub path for the default pruned model
        for this task
    :param default_quant_stub: sparsezoo stub path for the default quantized model
        for this task
    """

    pipeline_constructor: Callable[[Any], Pipeline]
    default_model_name: str
    base_stub: Optional[str] = None
    default_pruned_stub: Optional[str] = None
    default_quant_stub: Optional[str] = None


# Register all the supported tasks here
SUPPORTED_TASKS = {
    "ner": TaskInfo(
        pipeline_constructor=TokenClassificationPipeline,
        default_model_name="bert-base-uncased",
    ),
    "question-answering": TaskInfo(
        pipeline_constructor=QuestionAnsweringPipeline,
        default_model_name="bert-base-uncased",
        base_stub=(
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"
        ),
        default_pruned_stub=(
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/"
            "pruned-aggressive_98"
        ),
    ),
    "sentiment-analysis": TaskInfo(
        pipeline_constructor=TextClassificationPipeline,
        default_model_name="bert-base-uncased",
    ),
    "text-classification": TaskInfo(
        pipeline_constructor=TextClassificationPipeline,
        default_model_name="bert-base-uncased",
    ),
    "token-classification": TaskInfo(
        pipeline_constructor=TokenClassificationPipeline,
        default_model_name="bert-base-uncased",
    ),
}

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]


def pipeline(
    task: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    engine_type: str = DEEPSPARSE_ENGINE,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    max_length: int = 128,
    num_cores: Optional[int] = None,
    scheduler: Optional[str] = None,
    batch_size: Optional[int] = 1,
    **kwargs,
) -> Pipeline:
    """
    Utility factory method to build a Pipeline

    :param task: name of the task to define which pipeline to create. Currently
        supported task - "question-answering"
    :param model_name: canonical name of the hugging face model this model is based on
    :param model_path: path to model directory containing `model.onnx`, `config.json`,
        and `tokenizer.json` files, ONNX model file, or SparseZoo stub
    :param engine_type: inference engine name to use. Supported options are 'deepsparse'
        and 'onnxruntime'
    :param config: huggingface model config, if none provided, default will be used
        which will be from the model name or sparsezoo stub if given for model path
    :param tokenizer: huggingface tokenizer, if none provided, default will be used
    :param max_length: maximum sequence length of model inputs. default is 128
    :param num_cores: number of CPU cores to run engine with. Default is the maximum
        available
    :param scheduler: The scheduler to use for the engine. Can be None, single or multi.
    :param batch_size: The batch_size to use for the pipeline. Defaults to 1
        Note: `question-answering` pipeline only supports a batch_size of 1.
    :param kwargs: additional key word arguments for task specific pipeline constructor
    :return: Pipeline object for the given taks and model
    """

    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError(
            f"Unknown task {task}, available tasks are {list(SUPPORTED_TASKS.keys())}"
        )
    if engine_type not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unsupported engine {engine_type}, supported engines "
            f"are {SUPPORTED_ENGINES}"
        )
    if task == "question-answering" and batch_size != 1:
        raise ValueError(
            f"{task} pipeline only supports batch_size 1. "
            f"Supplied batch_size = {batch_size}"
        )
    task_info = SUPPORTED_TASKS[task]

    model_path = model_path or _get_default_model_path(task_info)
    model_name = model_name or task_info.default_model_name

    onnx_path, config_path, tokenizer_path = get_onnx_path_and_configs(model_path)

    # default the tokenizer and config to file in model directory or given model name
    config = config or config_path or model_name
    tokenizer = tokenizer or tokenizer_path or model_name

    # create model
    model, input_names = _create_model(
        onnx_path,
        engine_type,
        num_cores,
        max_length,
        scheduler=scheduler,
        batch_size=batch_size,
    )

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer_kwargs = tokenizer[1]
            tokenizer_kwargs["model_max_length"] = max_length
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, model_max_length=max_length
            )

    # Instantiate config if needed
    if config is not None and isinstance(config, str):
        config = AutoConfig.from_pretrained(config, finetuning_task=task)

    return task_info.pipeline_constructor(
        model=model,
        tokenizer=tokenizer,
        config=config,
        engine_type=engine_type,
        max_length=max_length,
        input_names=input_names,
        **kwargs,
    )


def _get_default_model_path(task_info: TaskInfo) -> str:
    if cpu.cpu_vnni_compatible() and task_info.default_quant_stub:
        return task_info.default_quant_stub
    return task_info.default_pruned_stub or task_info.base_stub


def _create_model(
    model_path: str,
    engine_type: str,
    num_cores: Optional[int],
    max_length: int = 128,
    scheduler: Optional[str] = None,
    batch_size: int = 1,
) -> Tuple[Union[Engine, "onnxruntime.InferenceSession"], List[str]]:
    onnx_path, input_names, _ = overwrite_transformer_onnx_model_inputs(
        model_path, max_length=max_length
    )

    if engine_type == DEEPSPARSE_ENGINE:
        model = compile_model(
            onnx_path,
            batch_size=batch_size,
            num_cores=num_cores,
            scheduler=scheduler,
        )
    elif engine_type == ORT_ENGINE:
        _validate_ort_import()
        sess_options = onnxruntime.SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        model = onnxruntime.InferenceSession(onnx_path, sess_options=sess_options)

    return model, input_names


def _validate_ort_import():
    if ort_import_error is not None:
        raise ImportError(
            "An exception occurred when importing onxxruntime. Please verify that "
            "onnxruntime is installed in order to use the onnxruntime inference "
            f"engine. \n\nException info: {ort_import_error}"
        )


def process_dataset(
    pipeline_object: Callable,
    data_path: str,
    batch_size: int,
    task: str,
    output_path: str,
) -> None:
    """
    :param pipeline_object: An instantiated pipeline Callable object
    :param data_path: Path to input file, supports csv, json and text files
    :param batch_size: batch_size to use for inference
    :param task: The task pipeline is instantiated for
    :param output_path: Path to a json file to output inference results to
    """
    batch_loader = get_batch_loader(
        data_file=data_path,
        batch_size=batch_size,
        task=task,
    )
    # Wraps pipeline object to make numpy types serializable
    pipeline_object = fix_numpy_types(pipeline_object)
    with open(output_path, "a") as output_file:
        for batch in batch_loader:
            batch_output = pipeline_object(**batch)
            json.dump(batch_output, output_file)
            output_file.write("\n")
