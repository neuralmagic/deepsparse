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

from typing import Any, Dict, List, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = ["TextGenerationPipeline"]


class TextGenerationInput(BaseModel):
    """
    Schema for inputs to text_classification pipelines
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to "
        "text generation task"
    )


class TextGenerationOutput(BaseModel):
    """
    Schema for inputs to text_classification pipelines
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing output from "
        "text generation task. This would be the the initial input extended "
        "by the next word predicted by the model."
    )
    is_done: Union[List[List[bool]], List[bool], bool] = Field(
        description="A boolean flag that indicates whether the model has "
        "finished generating the text."
    )


@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen"],
)
class TextGenerationPipeline(TransformersPipeline):
    """
    Transformers text generation pipeline.
    It takes a (batch of) text input of any length and generates a text output.
    The maximum length of the generated sequence is determined by the attribute
    `maximum_generation_length` of the model. The generation process terminate earlier
    if all the sequences in the batch terminate with the end of sequence token.

    example instantiation:
    ```python
    text_classifier = Pipeline.create(
        task="text_classification",
        model_path="text_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```
    example usage:
    ```python
    sequences = [["def hello_world():"], ["class FinancialPipeline(Pipeline):"]]
    OR
    sequences = ["def hello_world():", "class FinancialPipeline(Pipeline):"]
    OR
    sequences = "def hello_world():"

    out = pipeline(sequences = sequences)
    print(out.sequences)
    ```
    :param maximum_generation_length: The maximum length of the generated sequence.
        This means that the length of the resulting output will be a sum of the
        length of the input and the maximum_generation_length.
        The default is set to 256.
    """

    def __init__(
        self,
        *,
        maximum_generation_length: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.maximum_generation_length = maximum_generation_length

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        pydantic schema for inputs to this pipeline
        """
        return TextGenerationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        pydantic schema for outputs from this pipeline
        """
        return TextGenerationOutput

    def __call__(self, *args, **kwargs) -> BaseModel:
        """
        The main entry point for the pipeline. It runs
        prediction of the next token in the autoregressive
        fashion, with the possibility of earlier termination
        when end of sequence tokens are encountered for all the batches
        """
        for iteration in range(self.maximum_generation_length):
            # this is also the part of the code that could potentially
            # store KV lookups for the next iteration
            output = super().__call__(*args, **kwargs)
            if all(output.is_done):
                return output
            kwargs = dict(sequences=output.sequences)

        return output

    def process_inputs(
        self, inputs: BaseModel
    ) -> Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            TextGenerationInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        sequences = inputs.sequences
        if isinstance(sequences, List) and all(
            isinstance(sequence, List) and len(sequence) == 1 for sequence in sequences
        ):
            # if batch items contain only one sequence but are wrapped in lists, unwrap
            # for use as tokenizer input
            sequences = [sequence[0] for sequence in sequences]

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        tokens = self.tokenizer(
            sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
        )
        engine_input = self.tokens_to_engine_input(tokens)

        # a boolean mask that indicates which tokens are valid (are non-padding tokens)
        valid_tokens_mask = numpy.where(
            engine_input[0] == self.tokenizer.pad_token_id, 1, 0
        )

        preprocessing_kwargs = dict(
            input_sequence=engine_input[0], valid_tokens_mask=valid_tokens_mask
        )

        return engine_input, preprocessing_kwargs

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        :param engine_outputs: outputs of the pipeline engine. Must be a list of
            numpy arrays
        :param kwargs: additional keyword arguments to pass to the pipeline
        :return: outputs of this model embedded in a TextGenerationOutput object
        """

        input_sequence = kwargs.get("input_sequence", None)
        valid_tokens_mask = kwargs.get("valid_tokens_mask", None)

        if input_sequence is None:
            raise ValueError(
                "Expected input_sequence to be passed as a keyword argument. "
                "This is required for text generation pipeline."
            )
        if valid_tokens_mask is None:
            raise ValueError(
                "Expected valid_tokens_mask to be passed as a keyword argument. "
                "This is required for text generation pipeline."
            )

        # a list of booleans that indicates whether the sequence in the batch is done
        batch_num = engine_outputs[0].shape[0]
        is_done = [False] * batch_num

        if len(engine_outputs) == 41:
            # onnx model comes from sparseml.transformers.export
            logits = engine_outputs[0]  # logits are (Batch, Sequence_Len, Vocab_Size)
        elif len(engine_outputs) == 1:
            # onnx model comes from transformers.convert_graph_to_onnx
            logits = engine_outputs[0]
        else:
            raise ValueError(
                "Expected engine_outputs to be a list of length 1 or 41. "
                "This is the current assumption for the text generation pipeline."
            )

        # Using the mask to keep the valid tokens only
        valid_tokens = numpy.ma.masked_array(input_sequence, valid_tokens_mask)
        for batch_idx, valid_tokens_sequence in enumerate(valid_tokens):
            # by counting the number of valid tokens,
            # we can get the index of the last valid token
            # Is this assumption always valid?
            last_valid_token_idx = numpy.ma.count(valid_tokens_sequence)
            # get the logits that emerge after processing the last valid token
            last_logits = logits[batch_idx, last_valid_token_idx - 1, :]
            next_token = numpy.argmax(last_logits)
            if last_valid_token_idx >= self.sequence_length:
                raise ValueError(
                    f"The set input sequence length {self.sequence_length} "
                    "is too short to generate the next token."
                    "The point has been reached where the autoregressive "
                    f"sequence has generated {self.sequence_length} tokens"
                    f"and will continue generating more. "
                    f"To solve the problem, either increase the `sequence_length` "
                    f"of the pipeline so that it exceeds the "
                    f"`maximum_generation_length`= {self.maximum_generation_length}, "
                    f"or try to reduce `maximum_generation_length`."
                )

            input_sequence[batch_idx, last_valid_token_idx] = next_token

            if next_token == self.tokenizer.eos_token_id:
                # if the next token is the end of sequence token,
                # then the sequence is done
                is_done[batch_idx] = True

        return TextGenerationOutput(
            sequences=self.tokenizer.batch_decode(
                input_sequence, skip_special_tokens=True
            ),
            is_done=is_done,
        )

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[TransformersPipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        tokenizer = pipelines[0].tokenizer
        tokens = tokenizer(
            input_schema.sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=False,
            truncation=False,
        )
        input_seq_len = max(map(len, tokens["input_ids"]))
        return TransformersPipeline.select_bucket_by_seq_len(input_seq_len, pipelines)
