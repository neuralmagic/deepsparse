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
with mnli models
"""


from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, List, Optional, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

# from deepsparse.transformers.pipelines.zero_shot_text_classification import (
#    ZeroShotTextClassificationImplementation,
# ) TODO
from deepsparse.engine import Context
from deepsparse.utils import numpy_softmax

from .imp import (
    ZeroShotTextClassificationImplementation,
    ZeroShotTextClassificationOutput,
)


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


class MnliTextClassificationInput(BaseModel):
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
        "labels to create an mnli hypothesis. If provided, overrides the template "
        "in the mnli config.",
        default=None,
    )
    multi_class: Optional[bool] = Field(
        description="True if class probabilities are independent, default False. "
        "If provided, overrides the multi_class value in the mnli config.",
        default=None,
    )


class MnliTextClassificationPipeline(ZeroShotTextClassificationImplementation):
    def __init__(
        self,
        model_path: str,
        model_config: Optional[Union[MnliTextClassificationConfig, dict]] = None,
        num_sequences: int = 1,
        labels: Optional[List] = None,
        batch_size: Optional[int] = None,
        context: Optional[Context] = None,
        **kwargs,
    ):
        self._config = self._parse_config(model_config)
        self._num_sequences = num_sequences
        self._labels = self._parse_labels(labels)

        if batch_size is not None and batch_size != num_sequences:
            raise ValueError(
                f"This pipeline requires that batch_size {batch_size} match "
                f"num_sequences {num_sequences} when no static labels are "
                f"provided"
            )

        # if dynamic labels
        if self._labels is None:
            # TODO: refactor when elastic scheduler implements work stealing
            if context is None:
                # num_streams is arbitrarily chosen to be any value >= 2
                context = Context(num_cores=None, num_streams=2)
                kwargs.update({"context": context})

            self._thread_pool = ThreadPoolExecutor(
                max_workers=context.num_streams or 2,
                thread_name_prefix="deepsparse.pipelines.zero_shot_text_classifier",
            )

            kwargs.update({"batch_size": num_sequences})

        # if static labels
        else:
            if batch_size is not None and batch_size != num_sequences * len(
                self._labels
            ):
                raise ValueError(
                    f"This pipeline requires that batch_size {batch_size} match "
                    f"num_sequences times the number labels "
                    f"{num_sequences * len(self._labels)} when static labels are "
                    f"provided"
                )
            kwargs.update({"batch_size": num_sequences * len(self._labels)})

        super().__init__(model_path=model_path, **kwargs)

    @property
    def config_schema(self):
        """
        TODO
        """
        return MnliTextClassificationConfig

    @property
    def input_schema(self):
        return MnliTextClassificationInput

    def _parse_labels(self, labels: Union[None, List[str], str]) -> List[str]:
        """
        If given a string of comma separated labels, parses values into a list

        :param labels: A string of comma separated labels or a list of labels
        :return: a list of labels, parsed if originally in string form
        """
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

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
        labels = self._labels or inputs.labels
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

        # check for incorrect number of sequences
        num_sequences = (
            1 if isinstance(inputs.sequences, str) else len(inputs.sequences)
        )
        if num_sequences != self._num_sequences:
            raise ValueError(
                f"number of sequences {num_sequences} must match the number of "
                f"sequences the pipeline was instantiated with {self._num_sequences}"
            )

        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError(
                "You must include at least one label and at least one sequence."
            )

        # check for invalid hypothesis template
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

        tokens = self.tokenizer(
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

        return self.tokens_to_engine_input(tokens), postprocessing_kwargs

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        **kwargs,
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the mnli
            engine forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        sequences = kwargs["sequences"]
        candidate_labels = kwargs["labels"]
        multi_class = kwargs["multi_class"]

        outputs = engine_outputs
        if isinstance(outputs, list):
            outputs = outputs[0]

        # reshape sequences
        num_sequences = 1 if isinstance(sequences, str) else len(sequences)
        reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

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

        # hack: negate scores to perform reversed sort
        sorted_indexes = numpy.argsort(-1 * scores, axis=1)
        labels = [
            numpy.array(candidate_labels)[sorted_indexes[i]].tolist()
            for i in range(num_sequences)
        ]
        label_scores = numpy.take_along_axis(scores, sorted_indexes, axis=1).tolist()

        return self.output_schema(
            sequences=sequences,
            labels=labels,
            scores=label_scores,
        )

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        :param engine_inputs: list of numpy inputs to Pipeline engine forward
            pass
        :return: result of forward pass to Pipeline engine
        """
        # engine_inputs.shape: [transformer_inputs (3), num_labels * num_seqs, seq_len]
        # execute in parallel threads (dynamic labels) or as one batch (static labels)
        if self._labels is None:

            def _engine_forward(batch_index: int, batch_origin: int):
                labelwise_inputs = engine_inputs_numpy[
                    :, batch_origin : batch_origin + self._batch_size, :
                ]
                labelwise_inputs = [
                    labelwise_input for labelwise_input in labelwise_inputs
                ]
                engine_output = self.engine(labelwise_inputs)
                engine_outputs[batch_index] = engine_output

            engine_inputs_numpy = numpy.array(engine_inputs)
            engine_outputs = [
                None for _ in range(engine_inputs_numpy.shape[1] // self._batch_size)
            ]

            futures = [
                self._thread_pool.submit(_engine_forward, batch_index, batch_origin)
                for batch_index, batch_origin in enumerate(
                    range(0, engine_inputs_numpy.shape[1], self._batch_size)
                )
            ]
            wait(futures)
        else:
            engine_outputs = self.engine(engine_inputs)

        engine_outputs = numpy.array(engine_outputs)
        return engine_outputs

    @classmethod
    def get_current_sequence_length(input_schema: BaseModel) -> int:
        """
        Helper function to get max sequence length in provided sequences input

        :param input_schema: input to pipeline
        :return: max sequence length in input_schema
        """
        if isinstance(input_schema.sequences, str):
            current_seq_len = len(input_schema.sequences.split())
        elif isinstance(input_schema.sequences, list):
            current_seq_len = float("-inf")
            for _input in input_schema.sequences:
                if isinstance(_input, str):
                    current_seq_len = max(len(_input.split()), current_seq_len)
                elif isinstance(_input, list):
                    current_seq_len = max(
                        *(len(__input.split()) for __input in _input), current_seq_len
                    )
        else:
            raise ValueError(
                "Expected a str or List[str] or List[List[str]] as input but got "
                f"{type(input_schema.sequences)}"
            )

        return current_seq_len
