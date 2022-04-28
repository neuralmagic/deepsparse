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
Pipeline implementation and pydantic models for token classification transformers
tasks
"""


from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.file_utils import ExplicitEnum
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "AggregationStrategy",
    "TokenClassificationInput",
    "TokenClassificationResult",
    "TokenClassificationOutput",
    "TokenClassificationPipeline",
]


class AggregationStrategy(ExplicitEnum):
    """
    Valid aggregation strategies for postprocessing in the TokenClassificationPipeline
    """

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


class TokenClassificationInput(BaseModel):
    """
    Schema for inputs to token_classification pipelines
    """

    inputs: Union[List[str], str] = Field(
        description=(
            "A string or List of batch of strings representing input(s) to"
            "a token_classification task"
        )
    )


class TokenClassificationResult(BaseModel):
    """
    Schema for a classification of a single token
    """

    entity: str = Field(description="entity predicted for that token/word")
    score: float = Field(description="The corresponding probability for `entity`")
    index: int = Field(description="index of the corresponding token in the sentence")
    word: str = Field(description="token/word classified")
    start: Optional[int] = Field(
        description=(
            "index of the start of the corresponding entity in the sentence. "
            "Only exists if the offsets are available within the tokenizer"
        )
    )
    end: Optional[int] = Field(
        description=(
            "index of the end of the corresponding entity in the sentence. "
            "Only exists if the offsets are available within the tokenizer"
        )
    )
    is_grouped: bool = Field(
        default=False,
        description="True if this result is part of an entity group",
    )


class TokenClassificationOutput(BaseModel):
    """
    Schema for results of TokenClassificationPipeline inference. Classifications of each
    token stored in a list of lists of batch[sentence[token]]
    """

    predictions: List[List[TokenClassificationResult]] = Field(
        description=(
            "list of list of results of token classification pipeline. Outer list "
            "has one item for each sequence in the batch. Inner list has one "
            "TokenClassificationResult item per token in the given sequence"
        )
    )


@Pipeline.register(
    task="token_classification",
    task_aliases=["ner"],
)
class TokenClassificationPipeline(TransformersPipeline):
    """
    transformers token classification pipeline

    example instantiation:
    ```python
    token_classifier = Pipeline.create(
        task="token_classification",
        model_path="token_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```

    :param model_path: sparsezoo stub to a transformers model, an ONNX file, or
        (preferred) a directory containing a model.onnx, tokenizer config, and model
        config. If no tokenizer and/or model config(s) are found, then they will be
        loaded from huggingface transformers using the `default_model_name` key
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: static batch size to use for inference. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param sequence_length: sequence length to compile model and tokenizer for.
        Default is 128
    :param default_model_name: huggingface transformers model name to use to
        load a tokenizer and model config when none are provided in the `model_path`.
        Default is 'bert-base-uncased'
    :param aggregation_strategy: how to aggregate tokens in postprocessing. Options
        include 'none', 'simple', 'first', 'average', and 'max'. Default is None
    """

    def __init__(
        self,
        *,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.NONE,
        **kwargs,
    ):

        if isinstance(aggregation_strategy, str):
            aggregation_strategy = aggregation_strategy.strip().lower()
        self._aggregation_strategy = AggregationStrategy(aggregation_strategy)

        super().__init__(**kwargs)

    @property
    def aggregation_strategy(self) -> str:
        """
        :return: how to aggregate tokens in postprocessing. Options
            include 'none', 'simple', 'first', 'average', and 'max'
        """
        return self._aggregation_strategy.value

    @property
    def input_model(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return TokenClassificationInput

    @property
    def output_model(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return TokenClassificationOutput

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        """
        :param args: ordered arguments to pipeline, only an input_model object
            is supported as an arg for this function
        :param kwargs: keyword arguments to pipeline
        :return: pipeline arguments parsed into the given `input_model`
            schema if necessary. If an instance of the `input_model` is provided
            it will be returned
        """
        if args and kwargs:
            raise ValueError(
                f"{self.__class__} only support args OR kwargs. Found "
                f" {len(args)} args and {len(kwargs)} kwargs"
            )

        if args:
            if len(args) == 1:
                # passed input_model schema directly
                if isinstance(args[0], self.input_model):
                    return args[0]
                return self.input_model(inputs=args[0])
            else:
                return self.input_model(inputs=args)

        return self.input_model(**kwargs)

    def process_inputs(
        self,
        inputs: TokenClassificationInput,
    ) -> Tuple[List[numpy.ndarray], Dict[str, Any]]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            TokenClassificationInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
            and dictionary containing offset mappings and special tokens mask to
            be used during postprocessing
        """
        tokens = self.tokenizer(
            inputs.inputs,
            return_tensors="np",
            truncation=TruncationStrategy.LONGEST_FIRST.value,
            padding=PaddingStrategy.MAX_LENGTH.value,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )

        offset_mapping = (
            tokens.pop("offset_mapping")
            if self.tokenizer.is_fast
            else [None] * len(inputs.inputs)
        )
        special_tokens_mask = tokens.pop("special_tokens_mask")
        postprocessing_kwargs = dict(
            inputs=inputs,
            tokens=tokens,
            offset_mapping=offset_mapping,
            special_tokens_mask=special_tokens_mask,
        )

        return self.self.tokens_to_engine_input(tokens), postprocessing_kwargs

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        **kwargs,
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_model`
            format of this pipeline
        """
        inputs = kwargs["inputs"]
        tokens = kwargs["tokens"]
        offset_mapping = kwargs["offset_mapping"]
        special_tokens_mask = kwargs["special_tokens_mask"]

        predictions = []  # type: List[List[TokenClassificationResult]]

        for entities_index, current_entities in enumerate(engine_outputs[0]):
            input_ids = tokens["input_ids"][entities_index]

            scores = numpy.exp(current_entities) / numpy.exp(current_entities).sum(
                -1, keepdims=True
            )
            pre_entities = self._gather_pre_entities(
                inputs[entities_index],
                input_ids,
                scores,
                offset_mapping[entities_index],
                special_tokens_mask[entities_index],
            )
            grouped_entities = self._aggregate(pre_entities)
            # Filter anything that is in self.ignore_labels
            current_results = []  # type: List[TokenClassificationResult]
            for entity in grouped_entities:
                if entity.get("entity") not in self.ignore_labels or (
                    entity.get("entity_group") not in self.ignore_labels
                ):
                    continue
                if entity.get("entity_group"):
                    entity["entity"] = entity["entity_group"]
                    entity["is_grouped"] = True
                    del entity["entity_group"]
                current_results.append(TokenClassificationResult(**entity))
            predictions.append(current_results)

        return self.output_model(predictions=predictions)

    # utilities below adapted from transformers

    def _gather_pre_entities(
        self,
        sentence: str,
        input_ids: numpy.ndarray,
        scores: numpy.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: numpy.ndarray,
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

    def _aggregate(self, pre_entities: List[dict]) -> List[dict]:
        if self._aggregation_strategy in {
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
            entities = self._aggregate_words(pre_entities)

        if self._aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self._group_entities(entities)

    def _aggregate_word(self, entities: List[dict]) -> dict:
        word = self.tokenizer.convert_tokens_to_string(
            [entity["word"] for entity in entities]
        )
        if self._aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif self._aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.id2label[idx]
        elif self._aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = numpy.stack([entity["scores"] for entity in entities])
            average_scores = numpy.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError(
                f"Invalid aggregation_strategy: {self._aggregation_strategy}"
            )
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def _aggregate_words(self, entities: List[dict]) -> List[dict]:
        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self._aggregate_word(word_group))
                word_group = [entity]
        # Last item
        word_entities.append(self._aggregate_word(word_group))
        return word_entities

    def _group_sub_entities(self, entities: List[dict]) -> dict:
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = numpy.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": numpy.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def _get_tag(self, entity_name: str) -> Tuple[str, str]:
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

    def _group_entities(self, entities: List[dict]) -> List[dict]:

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
            bi, tag = self._get_tag(entity["entity"])
            last_bi, last_tag = self._get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self._group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self._group_sub_entities(entity_group_disagg))

        return entity_groups
