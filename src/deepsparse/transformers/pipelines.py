# this code is taken and adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines.py

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

import csv
import json
import os
import pickle
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from os.path import abspath, exists
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import onnx
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_all_providers,
)

from deepsparse import Engine, compile_model, cpu
from psutil import cpu_count
from sparsezoo import Zoo
from transformers.configuration_utils import PretrainedConfig
from transformers.convert_graph_to_onnx import (
    convert_pytorch,
    convert_tensorflow,
    infer_shapes,
)
from transformers.data import SquadExample, squad_convert_examples_to_features
from transformers.file_utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
)
from transformers.models.auto import AutoConfig, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.utils import logging


__all__ = ["Pipeline", "QuestionAnsweringPipeline", "pipeline"]

ONNX_CACHE_DIR = Path(os.path.dirname(__file__)).parent.joinpath(".onnx")
MAX_LENGTH = 128

logger = logging.get_logger(__name__)

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"


class PipelineException(Exception):
    """
    Raised by a :class:`~transformers.Pipeline` when handling __call__.

    Args:
        task (:obj:`str`): The task of the pipeline.
        model (:obj:`str`): The model used by the pipeline.
        reason (:obj:`str`): The error message to display.
    """

    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each :class:`~transformers.pipelines.Pipeline`.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultArgumentHandler(ArgumentHandler):
    """
    Default argument parser handling parameters for each :class:`~transformers.pipelines.Pipeline`.
    """

    @staticmethod
    def handle_kwargs(kwargs: Dict) -> List:
        if len(kwargs) == 1:
            output = list(kwargs.values())
        else:
            output = list(chain(kwargs.values()))

        return DefaultArgumentHandler.handle_args(output)

    @staticmethod
    def handle_args(args: Sequence[Any]) -> List[str]:

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

            # If not instance of list, then it should instance of iterable
            elif isinstance(args, Iterable):
                return list(chain.from_iterable(chain(args)))
            else:
                raise ValueError(
                    "Invalid input type {}. Pipeline supports Union[str, Iterable[str]]".format(
                        type(args)
                    )
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


PIPELINE_INIT_ARGS = r"""
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no
            model is provided.
        task (:obj:`str`, defaults to :obj:`""`):
            A task-identifier for the pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model
            on the associated CUDA device id.
        binary_output (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
"""


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations.
    Pipeline workflow is defined as a sequence of the following operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument or using onnx runtime (see below).

    Some pipeline, like for instance :class:`~transformers.FeatureExtractionPipeline` (:obj:`'feature-extraction'` )
    output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the :obj:`binary_output` constructor argument. If set to :obj:`True`, the output will be stored in the
    pickle format.
    """

    default_input_names = None

    def __init__(
        self,
        model: Union[Engine, InferenceSession],
        tokenizer: PreTrainedTokenizer,
        config: PretrainedConfig,
        engine: str,
        task: str = "",
        input_names: Optional[List[str]] = None,
        args_parser: ArgumentHandler = None,
        binary_output: bool = False,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.engine = engine
        self.task = task
        self.input_names = input_names
        self.binary_output = binary_output
        self._args_parser = args_parser or DefaultArgumentHandler()
        self._framework = (
            "np" if self.engine in [DEEPSPARSE_ENGINE, ORT_ENGINE] else "pt"
        )

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def _parse_and_tokenize(
        self, *args, padding=True, add_special_tokens=True, **kwargs
    ):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self._framework,
            padding=padding,
        )

        return inputs

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching.
        Args:
            inputs: dict holding all the keyworded arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array.
        Returns:
            Numpy array
        """

        if self.engine == ORT_ENGINE:

            # TODO: filter by valid name
            #  inputs = {k: v for k, v in inputs.items() if k in self.input_names}
            return self.model.run(None, dict(zip(self.input_names, inputs.values())))
        elif self.engine == DEEPSPARSE_ENGINE:
            return self.model.run(list(inputs.values()))
        # TODO: torch
        # with self.device_placement():
        #         with torch.no_grad():
        #             inputs = self.ensure_tensor_on_device(**inputs)
        #             predictions = self.model(**inputs)[0].cpu()
        # if return_tensors:
        #     return predictions
        # else:
        #     return predictions.numpy()


class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped
    to internal :class:`~transformers.SquadExample`.

    QuestionAnsweringArgumentHandler manages all the possible to create a :class:`~transformers.SquadExample` from
    the command-line supplied arguments.
    """

    def __call__(self, *args, **kwargs):
        # Position args, handling is sensibly the same as X and data, so forwarding to avoid duplicating
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
                            "You need to provide a dictionary with keys {question:..., context:...}"
                        )

                    inputs[i] = QuestionAnsweringPipeline.create_sample(**item)

                elif not isinstance(item, SquadExample):
                    raise ValueError(
                        "{} argument needs to be of type (list[SquadExample | dict], SquadExample, dict)".format(
                            "X" if "X" in kwargs else "data"
                        )
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
            raise ValueError("Unknown arguments {}".format(kwargs))

        if not isinstance(inputs, list):
            inputs = [inputs]

        return inputs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using any :obj:`ModelForQuestionAnswering`. See the
    `question answering examples <../task_summary.html#question-answering>`__ for more information.

    This question answering pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=question-answering>`__.
    """

    default_input_names = "question,context"

    def __init__(
        self,
        model: Union[Engine, InferenceSession],
        tokenizer: PreTrainedTokenizer,
        engine: str,
        task: str = "",
        input_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            engine=engine,
            args_parser=QuestionAnsweringArgumentHandler(),
            task=task,
            input_names=input_names,
            **kwargs,
        )

    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the :class:`~transformers.SquadExample` internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to
        :class:`~transformers.SquadExample`.

        We currently support extractive question answering.

        Arguments:
            question (:obj:`str` or :obj:`List[str]`): The question(s) asked.
            context (:obj:`str` or :obj:`List[str]`): The context(s) in which we will look for the answer.

        Returns:
            One or a list of :class:`~transformers.SquadExample`: The corresponding
            :class:`~transformers.SquadExample` grouping question and context.
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

        Args:
            args (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`):
                One or several :class:`~transformers.SquadExample` containing the question and context.
            X (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context
                (will be treated the same way as if passed as the first positional argument).
            data (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context
                (will be treated the same way as if passed as the first positional argument).
            question (:obj:`str` or :obj:`List[str]`):
                One or several question(s) (must be used in conjunction with the :obj:`context` argument).
            context (:obj:`str` or :obj:`List[str]`):
                One or several context(s) associated with the qustion(s) (must be used in conjunction with the
                :obj:`question` argument).
            topk (:obj:`int`, `optional`, defaults to 1):
                The number of answers to return (will be chosen by order of likelihood).
            doc_stride (:obj:`int`, `optional`, defaults to 128):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            max_answer_len (:obj:`int`, `optional`, defaults to 15):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_seq_len (:obj:`int`, `optional`, defaults to 384):
                The maximum length of the total sentence (context + question) after tokenization. The context will be
                split in several chunks (using :obj:`doc_stride`) if needed.
            max_question_len (:obj:`int`, `optional`, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            handle_impossible_answer (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not we accept impossible as an answer.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the
            following keys:

            - **score** (:obj:`float`) -- The probability associated to the answer.
            - **start** (:obj:`int`) -- The start index of the answer (in the tokenized version of the input).
            - **end** (:obj:`int`) -- The end index of the answer (in the tokenized version of the input).
            - **answer** (:obj:`str`) -- The answer to the question.
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", MAX_LENGTH)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("handle_impossible_answer", False)

        if kwargs["topk"] < 1:
            raise ValueError(
                "topk parameter should be >= 1 (got {})".format(kwargs["topk"])
            )

        if kwargs["max_answer_len"] < 1:
            raise ValueError(
                "max_answer_len parameter should be >= 1 (got {})".format(
                    kwargs["max_answer_len"]
                )
            )

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
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
            # fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
            # start, end = self.model(**fw_args)[:2]
            # start, end = start.cpu().numpy(), end.cpu().numpy()

            min_null_score = 1000000  # large and positive
            answers = []
            for (feature, start_, end_) in zip(features, start, end):
                # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
                undesired_tokens = (
                    np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask
                )

                # Generate mask
                undesired_tokens_mask = undesired_tokens == 0.0

                # Make sure non-context indexes in the tensor cannot contribute to the softmax
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
                char_to_word = np.array(example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                answers += [
                    {
                        "score": score.item(),
                        "start": np.where(char_to_word == feature.token_to_orig_map[s])[
                            0
                        ][0].item(),
                        "end": np.where(char_to_word == feature.token_to_orig_map[e])[
                            0
                        ][-1].item(),
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

    def decode(
        self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int
    ) -> Tuple:
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and will generate probalities for each span to be
        the actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than
        max_answer_len or answer end position being before the starting position.
        The method supports output the k-best answer through the topk argument.

        Args:
            start (:obj:`np.ndarray`): Individual start probabilities for each token.
            end (:obj:`np.ndarray`): Individual end probabilities for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
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
        When decoding from token probalities, this method maps token indexes to actual word in
        the initial context.

        Args:
            text (:obj:`str`): The actual context to extract the answer from.
            start (:obj:`int`): The answer starting token index.
            end (:obj:`int`): The answer end token index.

        Returns:
            Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
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


@dataclass
class TaskInfo:
    pipeline_constructor: Callable[[Any], Pipeline]
    default_model_name: str
    base_stub: Optional[str] = None
    default_pruned_stub: Optional[str] = None
    default_quant_stub: Optional[str] = None


# Register all the supported tasks here
SUPPORTED_TASKS = {
    "question-answering": TaskInfo(
        pipeline_constructor=QuestionAnsweringPipeline,
        default_model_name="bert-base-uncased",
        base_stub="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none",
        default_pruned_stub="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate",
    )
}

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]


def pipeline(
    task: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    engine: str = DEEPSPARSE_ENGINE,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    num_cores: Optional[int] = None,
    num_sockets: Optional[int] = None,
    **kwargs,
) -> Pipeline:
    """
    Utility factory method to build a :class:`~transformers.Pipeline`.

    Pipelines are made of:

        - A :doc:`tokenizer <tokenizer>` in charge of mapping raw textual input to token.
        - A :doc:`model <model>` to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`: will return a :class:`~transformers.FeatureExtractionPipeline`.
            - :obj:`"sentiment-analysis"`: will return a :class:`~transformers.TextClassificationPipeline`.
            - :obj:`"ner"`: will return a :class:`~transformers.TokenClassificationPipeline`.
            - :obj:`"question-answering"`: will return a :class:`~transformers.QuestionAnsweringPipeline`.
            - :obj:`"zero-shot-classification"`: will return a :class:`~transformers.ZeroShotClassificationPipeline`.
        model (:obj:`str`, `optional`):
            The model that will be used by the pipeline to make predictions. This should be a model identifier

            If not provided, the default for the :obj:`task` will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If not provided, the default for the :obj:`task` will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default for the :obj:`task` will be loaded.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no
            model is provided.
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

        from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        # Sentiment analysis pipeline
        pipeline('sentiment-analysis')

        # Question answering pipeline, specifying the checkpoint identifier
        pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')
    """
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError(
            f"Unknown task {task}, available tasks are {list(SUPPORTED_TASKS.keys())}"
        )
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unsupported engine {engine}, supported engines are {SUPPORTED_ENGINES}"
        )

    task_info = SUPPORTED_TASKS[task]
    model_path = model_path or _get_default_model_path(task_info)
    model_name = model_name or task_info.default_model_name

    # default the tokenizer and config to given model name
    tokenizer = tokenizer or model_name
    config = config or model_name

    # create model
    if model_path.startswith("zoo:"):
        model_path = _download_zoo_model(model_path)
    model, input_names = _create_model(model_path, engine, num_cores, num_sockets)

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

    # Instantiate config if needed
    if config is not None and isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    return task_info.pipeline_constructor(
        model=model,
        tokenizer=tokenizer,
        task=task,
        config=config,
        engine=engine,
        input_names=input_names,
        **kwargs,
    )


def _get_default_model_path(task_info: TaskInfo) -> str:
    if cpu.cpu_vnni_compatible() and task_info.default_quant_stub:
        return task_info.default_quant_stub
    return task_info.default_pruned_stub or task_info.base_stub


def _download_zoo_model(model_path: str) -> str:
    model = Zoo.load_model_from_stub(model_path)
    return model.onnx_file.downloaded_path()


def _overwrite_model_inputs(
    path: str,
) -> Tuple[str, List[str], Optional[NamedTemporaryFile]]:
    # overwrite input shapes
    model = onnx.load(path)
    initializer_input_names = set([node.name for node in model.graph.initializer])
    external_inputs = [
        inp for inp in model.graph.input if inp.name not in initializer_input_names
    ]
    input_names = []
    for external_input in external_inputs:
        external_input.type.tensor_type.shape.dim[0].dim_value = 1
        external_input.type.tensor_type.shape.dim[1].dim_value = MAX_LENGTH
        input_names.append(external_input.name)

    # Save modified model
    tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
    onnx.save(model, tmp_file.name)

    return tmp_file.name, input_names, tmp_file


def _create_model(
    model_path: str,
    engine: str,
    num_cores: Optional[int],
    num_sockets: Optional[int],
) -> Tuple[Union[Engine, InferenceSession], List[str]]:
    onnx_path, input_names, _ = _overwrite_model_inputs(model_path)

    if engine == DEEPSPARSE_ENGINE:
        model = compile_model(
            onnx_path, batch_size=1, num_cores=num_cores, num_sockets=num_sockets
        )
    elif engine == ORT_ENGINE:
        sess_options = SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_model = onnx.load(onnx_path)
        model = InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )

    return model, input_names
