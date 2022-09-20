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

# postprocessing adapted from huggingface/transformers

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pipeline implementation and pydantic models for zero-shot text classification task
with mnli models
"""


from typing import List, Optional, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse.transformers.pipelines.zero_shot_text_classification import (
    ZeroShotTextClassificationInputBase,
    ZeroShotTextClassificationPipelineBase,
)
from deepsparse.utils import numpy_softmax


__all__ = [
    "MnliTextClassificationConfig",
    "MnliTextClassificationInput",
    "MnliTextClassificationPipeline",
]


class MnliTextClassificationConfig(BaseModel):
    """
    Schema for configuration options when running zero shot with mnli
    """

    hypothesis_template: str = Field(
        description="A formattable template for wrapping around the provided "
        "labels to create an mnli hypothesis",
        default="This text is about {}",
    )
    entailment_index: int = Field(
        description="Index of mnli model outputs which denotes entailment", default=0
    )
    contradiction_index: int = Field(
        description="Index of mnli model outputs which denotes contradiction", default=2
    )
    multi_class: bool = Field(
        description="True if class probabilities are independent, default False",
        default=False,
    )


class MnliTextClassificationInput(ZeroShotTextClassificationInputBase):
    """
    Schema for inputs to zero_shot_text_classification pipelines
    Each sequence and each candidate label must be paired and passed through
    the model, so the total number of forward passes is num_labels * num_sequences
    """

    labels: Optional[Union[List[str], str]] = Field(
        description="The set of possible class labels to classify each "
        "sequence into. Can be a single label, a string of comma-separated "
        "labels, or a list of labels."
    )
    hypothesis_template: Optional[str] = Field(
        description="A formattable template for wrapping around the provided "
        "labels to create an mnli hypothesis. If provided, overrides the template "
        "in the mnli config.",
        default=None,
    )
    multi_class: Optional[bool] = Field(
        description="True if class probabilities are independent, default False. "
        "If provided, overrides the multi_class value in the config.",
        default=None,
    )


class MnliTextClassificationPipeline(ZeroShotTextClassificationPipelineBase):
    def __init__(
        self,
        model_path: str,
        model_config: Optional[Union[MnliTextClassificationConfig, dict]] = None,
        labels: Optional[List] = None,
        **kwargs,
    ):
        self._config = self.parse_config(model_config)
        self._labels = self.parse_labels(labels)

        if (
            self._config.hypothesis_template is not None
            and self._config.hypothesis_template.format("sample_label")
            == self._config.hypothesis_template
        ):
            raise ValueError(
                "The provided hypothesis_template "
                f"`{self._config.hypothesis_template}` was not able to be formatted. "
                "Make sure the passed template includes formatting syntax such "
                "as `{}` where the label should go."
            )

        if self._config.entailment_index == self._config.contradiction_index:
            raise ValueError("entailment_index must differ from contradiction_index")

        if self._config.entailment_index > 2:
            raise ValueError("entailment_index must be less than or equal to 2")

        if self._config.contradiction_index > 2:
            raise ValueError("contradiction_index must be less than or equal to 2")

        super().__init__(model_path=model_path, **kwargs)

    @property
    def config_schema(self) -> Type[MnliTextClassificationConfig]:
        """
        Config schema the model_config argument must comply to
        """
        return MnliTextClassificationConfig

    @property
    def input_schema(self) -> Type[MnliTextClassificationInput]:
        """
        Input schema inputs using the mnli model scheme must comply to
        """
        return MnliTextClassificationInput

    def process_inputs(
        self,
        inputs: MnliTextClassificationInput,
    ) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        sequences = inputs.sequences
        labels = self.parse_labels(self._labels or inputs.labels)
        hypothesis_template = (
            inputs.hypothesis_template
            if inputs.hypothesis_template is not None
            else self._config.hypothesis_template
        )
        multi_class = (
            inputs.multi_class
            if inputs.multi_class is not None
            else self._config.multi_class
        )
        if isinstance(sequences, str):
            sequences = [sequences]

        # check for absent labels
        if inputs.labels is None and self._labels is None:
            raise ValueError(
                "You must provide either static labels during pipeline creation or "
                "dynamic labels at inference time"
            )

        # check for conflicting labels
        if inputs.labels is not None and self._labels is not None:
            raise ValueError(
                "Found both static labels and dynamic labels at inference time. You "
                "must provide only one"
            )

        # check batch size divides sequences * labels
        if (len(labels) * len(sequences)) % self._batch_size != 0:
            raise ValueError(
                "The number of sequences times the number of labels "
                f"({len(labels) * len(sequences)}) must be divisible by batch_size "
                f"{self._batch_size}"
            )

        # check for invalid hypothesis template
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                f"The provided hypothesis_template `{hypothesis_template}` was not "
                "able to be formatted with the target labels. Make sure the "
                "passed template includes formatting syntax such as {} where "
                "the label should go."
            )

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend(
                [[sequence, hypothesis_template.format(label)] for label in labels]
            )

        tokens = self.tokenizer(
            sequence_pairs,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.ONLY_FIRST.value,
        )  # tokenize only_first so that hypothesis (label) is not truncated

        postprocessing_kwargs = dict(
            sequences=inputs.sequences,  # do not include list wrapping
            candidate_labels=labels,
            multi_class=multi_class,
        )

        return self.tokens_to_engine_input(tokens), postprocessing_kwargs

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        sequences: Union[str, List[str]],
        candidate_labels: List[str],
        multi_class: bool,
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the mnli
            engine forward pass
        :param sequences: original sequences passed to inference
        :param candidate_labels: labels to match scores with
        :param multi_class: if True, calculate each class score independently from
            one another
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        # reshape sequences
        num_sequences = 1 if isinstance(sequences, str) else len(sequences)
        reshaped_outputs = numpy.reshape(
            engine_outputs, (num_sequences, len(candidate_labels), -1)
        )

        # calculate scores
        if not multi_class:
            entailment_logits = reshaped_outputs[:, :, self._config.entailment_index]
            scores = numpy_softmax(entailment_logits, axis=1)
        else:
            entailment_contradiction_logits = reshaped_outputs[
                :, :, [self._config.entailment_index, self._config.contradiction_index]
            ]
            probabilities = numpy_softmax(entailment_contradiction_logits, axis=2)
            scores = probabilities[:, :, 0]

        # negate scores to perform reversed sort
        sorted_indexes = numpy.argsort(scores * -1, axis=1)
        labels = [
            numpy.array(candidate_labels)[sorted_indexes[i]].tolist()
            for i in range(num_sequences)
        ]
        label_scores = numpy.take_along_axis(scores, sorted_indexes, axis=1).tolist()

        # reduce dims if passed a string, not a list of strings
        if isinstance(sequences, str):
            labels = labels[0]
            label_scores = label_scores[0]

        return self.output_schema(
            sequences=sequences,
            labels=labels,
            scores=label_scores,
        )
