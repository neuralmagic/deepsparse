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


from typing import List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse.transformers.pipelines.zero_shot_text_classification import (
    ZeroShotTextClassificationInputBase,
    ZeroShotTextClassificationOutput,
    ZeroShotTextClassificationPipelineBase,
)
from deepsparse.utils import numpy_softmax


__all__ = [
    "MnliTextClassificationConfig",
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
    labels: List[str] = Field(
        description="The set of possible class labels to classify each "
        "sequence into. Can be a single label, a string of comma-separated "
        "labels, or a list of labels."
    )


class _PostProcessingConfig(MnliTextClassificationConfig):
    inputs: List[ZeroShotTextClassificationInputBase]


class MnliTextClassificationPipeline(ZeroShotTextClassificationPipelineBase):
    def __init__(
        self,
        model_path: str,
        batch_size: int = 1,
        model_config: Optional[Union[MnliTextClassificationConfig, dict]] = None,
        **kwargs,
    ):
        self._config = self.parse_config(model_config)

        if self._labels and batch_size % len(self._labels) != 0:
            raise ValueError(
                f"if static labels are provided then batch_size {batch_size} must "
                f"be divisible by the number of labels {len(self._labels)}"
            )

        # will add support for batch_size == None when dynamic batch lands
        if not self._labels and batch_size != 1:
            raise ValueError(
                "if no static labels are provided then batch_size must be set to 1"
            )

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

        kwargs.update({"batch_size": batch_size})
        super().__init__(model_path=model_path, **kwargs)

    @property
    def config_schema(self) -> Type[MnliTextClassificationConfig]:
        """
        Config schema the model_config argument must comply to
        """
        return MnliTextClassificationConfig

    @property
    def input_schema(self) -> Type[ZeroShotTextClassificationInputBase]:
        """
        Input schema inputs using the mnli model scheme must comply to
        """
        return ZeroShotTextClassificationInputBase

    def parse_inputs(
        self,
        sequences: Union[List[str], str],
        labels: Optional[Union[List[str], str]] = None,
        hypothesis_template: Optional[str] = None,
        multi_class: Optional[bool] = None,
    ) -> Tuple[List[ZeroShotTextClassificationInputBase], MnliTextClassificationConfig]:
        if labels is None and self.config.labels is None:
            raise ValueError(
                "You must provide either static labels during pipeline creation or "
                "dynamic labels at inference time"
            )
        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self.parse_labels(self._labels or labels)
        hypothesis_template = hypothesis_template or self._config.hypothesis_template
        multi_class = multi_class or self._config.multi_class
        cfg = MnliTextClassificationConfig(
            hypothesis_template=hypothesis_template,
            multi_class=multi_class,
            labels=labels,
        )
        return [ZeroShotTextClassificationInputBase(sequence=s) for s in sequences], cfg

    def process_inputs(
        self,
        inputs: List[ZeroShotTextClassificationInputBase],
        cfg: MnliTextClassificationConfig,
    ) -> Tuple[List[numpy.ndarray], _PostProcessingConfig]:
        """
        :param inputs: inputs to the pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        sequences = [i.sequence for i in inputs]

        # check for correct size if static labels
        if self._labels and len(sequences) != self._batch_size // len(self._labels):
            raise ValueError(
                "If static labels are provided, then the number of sequences "
                f"{len(sequences)} must match batch_size divided by the number of "
                f"labels {self._batch_size // len(self._labels)}"
            )

        # check for invalid hypothesis template
        if cfg.hypothesis_template.format(cfg.labels[0]) == cfg.hypothesis_template:
            raise ValueError(
                f"The provided hypothesis_template `{cfg.hypothesis_template}` was not "
                "able to be formatted with the target labels. Make sure the "
                "passed template includes formatting syntax such as {} where "
                "the label should go."
            )

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend(
                [
                    [sequence, cfg.hypothesis_template.format(label)]
                    for label in cfg.labels
                ]
            )

        tokens = self.tokenizer(
            sequence_pairs,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            # tokenize only_first so that hypothesis (label) is not truncated
            truncation=TruncationStrategy.ONLY_FIRST.value,
        )

        return self.tokens_to_engine_input(tokens), _PostProcessingConfig(
            inputs=inputs, **cfg.dict()
        )

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        :param engine_inputs: list of numpy inputs to Pipeline engine forward pass
        :return: result of forward pass to Pipeline engine
        """
        engine_inputs_numpy = numpy.array(engine_inputs)
        if self._labels is None:
            engine_outputs = []
            for sample_i in range(engine_inputs_numpy.shape[1]):
                engine_input = engine_inputs_numpy[:, sample_i : (sample_i + 1), :]
                engine_input = [input for input in engine_input]
                engine_output = self.engine(engine_input)
                engine_outputs.append(engine_output)
            return engine_outputs
        else:
            return self.engine(engine_inputs)

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        cfg: _PostProcessingConfig,
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
        num_sequences = len(cfg.inputs)
        reshaped_outputs = numpy.reshape(
            engine_outputs, (num_sequences, len(cfg.labels), -1)
        )

        # calculate scores
        if not cfg.multi_class:
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
            numpy.array(cfg.labels)[sorted_indexes[i]].tolist()
            for i in range(num_sequences)
        ]
        label_scores = numpy.take_along_axis(scores, sorted_indexes, axis=1).tolist()

        return [
            ZeroShotTextClassificationOutput(
                sequence=input.sequence, label=label, score=score
            )
            for input, label, score in zip(cfg.inputs, labels, label_scores)
        ]
