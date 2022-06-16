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
Helper functions and pydantic models for zero-shot text classification task
with nli models
"""


from concurrent.futures import wait
from typing import TYPE_CHECKING, List, Optional, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse.utils import numpy_softmax


if TYPE_CHECKING:
    from deepsparse.transformers.pipelines.zero_shot_text_classification import (
        ZeroShotTextClassificationPipeline,
    )


__all__ = [
    "NliTextClassificationConfig",
    "NliTextClassificationInput",
    "process_nli_inputs",
    "process_nli_engine_outputs",
    "nli_engine_forward",
]


class NliTextClassificationConfig(BaseModel):
    """
    Schema for configuration options when running zero shot with nli
    """

    hypothesis_template: str = Field(
        description="A formattable template for wrapping around the provided "
        "labels to create an nli hypothesis",
        default="This text is about {}",
    )
    entailment_index: int = Field(
        description="Index of nli model outputs which denotes entailment", default=0
    )
    contradiction_index: int = Field(
        description="Index of nli model outputs which denotes contradiction", default=2
    )
    multi_class: bool = Field(
        description="True if class probabilities are independent, default False",
        default=False,
    )


class NliTextClassificationInput(BaseModel):
    """
    Schema for inputs to zero_shot_text_classification pipelines
    Each sequence and each candidate label must be paired and passed through
    the model, so the total number of forward passes is num_labels * num_sequences
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to "
        "zero_shot_text_classification task"
    )
    labels: Union[None, List[List[str]], List[str], str] = Field(
        description="The set of possible class labels to classify each "
        "sequence into. Can be a single label, a string of comma-separated "
        "labels, or a list of labels."
    )
    hypothesis_template: Optional[str] = Field(
        description="A formattable template for wrapping around the provided "
        "labels to create an nli hypothesis. If provided, overrides the template "
        "in the nli config.",
        default=None,
    )
    multi_class: Optional[bool] = Field(
        description="True if class probabilities are independent, default False. "
        "If provided, overrides the multi_class value in the nli config.",
        default=None,
    )


def process_nli_inputs(
    pipeline: "ZeroShotTextClassificationPipeline",
    inputs: NliTextClassificationInput,
    config: NliTextClassificationConfig,
) -> List[numpy.ndarray]:
    """
    :param pipeline: pipeline instance performing the input processing
    :param inputs: inputs to the pipeline
    :param config: instance of NliTextClassificationConfig
    :return: inputs of this model processed into a list of numpy arrays that
        can be directly passed into the forward pass of the pipeline engine
    """
    sequences = inputs.sequences
    labels = pipeline._labels or inputs.labels
    hypothesis_template = (
        inputs.hypothesis_template
        if inputs.hypothesis_template is not None
        else config.hypothesis_template
    )
    multi_class = (
        inputs.multi_class if inputs.multi_class is not None else config.multi_class
    )

    if len(labels) == 0 or len(sequences) == 0:
        raise ValueError(
            "You must include at least one label and at least one sequence."
        )

    if isinstance(sequences, str):
        sequences = [sequences]

    labels = pipeline._parse_labels(labels)

    if hypothesis_template.format(labels[0]) == hypothesis_template:
        raise ValueError(
            (
                'The provided hypothesis_template "{}" was not able to be '
                "formatted with the target labels. Make sure the passed template "
                "includes formatting syntax such as {{}} where the label should go."
            ).format(hypothesis_template)
        )

    sequence_pairs = []
    for sequence in sequences:
        sequence_pairs.extend(
            [[sequence, hypothesis_template.format(label)] for label in labels]
        )

    tokens = pipeline.tokenizer(
        sequence_pairs,
        add_special_tokens=True,
        return_tensors="np",
        padding=PaddingStrategy.MAX_LENGTH.value,
        # tokenize only_first so that hypothesis (label) is not truncated
        truncation=TruncationStrategy.ONLY_FIRST.value,
    )

    postprocessing_kwargs = dict(
        sequences=sequences,
        labels=labels,
        multi_class=multi_class,
    )

    return pipeline.tokens_to_engine_input(tokens), postprocessing_kwargs


def process_nli_engine_outputs(
    pipeline: "ZeroShotTextClassificationPipeline",
    engine_outputs: List[numpy.ndarray],
    config: NliTextClassificationConfig,
    **kwargs,
) -> BaseModel:
    """
    :param pipeline: pipeline instance performing the input processing
    :param engine_outputs: list of numpy arrays that are the output of the nli
        engine forward pass
    :param config: instance of NliTextClassificationConfig
    :return: outputs of engine post-processed into an object in the `output_schema`
        format of this pipeline
    """
    sequences = kwargs["sequences"]
    candidate_labels = kwargs["labels"]
    multi_class = kwargs["multi_class"]

    outputs = engine_outputs
    if isinstance(outputs, list):
        outputs = outputs[0]

    # Reshape sequences
    num_sequences = 1 if isinstance(sequences, str) else len(sequences)
    reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

    # Calculate scores
    if not multi_class:
        entailment_logits = reshaped_outputs[:, :, config.entailment_index]
        scores = numpy_softmax(entailment_logits, axis=1)
    else:
        entailment_contradiction_logits = reshaped_outputs[
            :, :, [config.entailment_index, config.contradiction_index]
        ]
        probabilities = numpy_softmax(entailment_contradiction_logits, axis=2)
        scores = probabilities[:, :, 0]

    # Hack: negate scores to perform reversed sort
    sorted_indexes = numpy.argsort(-1 * scores, axis=1)
    labels = [
        numpy.array(candidate_labels)[sorted_indexes[i]].tolist()
        for i in range(num_sequences)
    ]
    label_scores = numpy.take_along_axis(scores, sorted_indexes, axis=1).tolist()

    return pipeline.output_schema(
        sequences=sequences,
        labels=labels,
        scores=label_scores,
    )


def nli_engine_forward(
    pipeline: "ZeroShotTextClassificationPipeline", engine_inputs: List[numpy.ndarray]
) -> List[numpy.ndarray]:
    """
    :param pipeline: pipeline instance performing the input processing
    :param engine_inputs: list of numpy inputs to Pipeline engine forward
        pass
    :return: result of forward pass to Pipeline engine
    """
    # engine_inputs.shape: [transformer_inputs (3), num_labels * num_seqs, seq_len]
    # Execute in parallel threads (dynamic labels) or as one batch (static labels)
    if pipeline._labels is None:

        def _engine_forward(batch_index: int, batch_origin: int):
            labelwise_inputs = engine_inputs_numpy[
                :, batch_origin : batch_origin + pipeline._batch_size, :
            ]
            labelwise_inputs = [labelwise_input for labelwise_input in labelwise_inputs]
            engine_output = pipeline.engine(labelwise_inputs)
            engine_outputs[batch_index] = engine_output

        engine_inputs_numpy = numpy.array(engine_inputs)
        engine_outputs = [
            None for _ in range(engine_inputs_numpy.shape[1] // pipeline._batch_size)
        ]

        futures = [
            pipeline._thread_pool.submit(_engine_forward, batch_index, batch_origin)
            for batch_index, batch_origin in enumerate(
                range(0, engine_inputs_numpy.shape[1], pipeline._batch_size)
            )
        ]
        wait(futures)
    else:
        engine_outputs = pipeline.engine(engine_inputs)

    engine_outputs = numpy.array(engine_outputs)
    return engine_outputs
