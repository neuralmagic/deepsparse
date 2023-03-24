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

import os
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy
import onnx
from pydantic import BaseModel, Field
from transformers import BatchEncoding
from transformers.models.auto import AutoConfig, AutoTokenizer

from deepsparse import Pipeline
from deepsparse.transformers.helpers import get_onnx_path_and_configs
from deepsparse.transformers.pipelines import TransformersPipeline
from scipy.special import softmax


_MODEL_DIR_ONNX_MULTI_TOKEN_NAME = "decoder_model.onnx"
_MODEL_DIR_ONNX_NAME = "model.onnx"

__all__ = ["TextGenerationPipeline"]


def overwrite_transformer_onnx_model_inputs(
    path: str,
    batch_size: int = 1,
    max_length: int = 128,
    output_path: Optional[str] = None,
) -> Tuple[Optional[str], List[str], Optional[NamedTemporaryFile]]:
    """
    Overrides an ONNX model's inputs to have the given batch size and sequence lengths.
    Assumes that these are the first and second shape indices of the given model inputs
    respectively

    :param path: path to the ONNX model to override
    :param batch_size: batch size to set
    :param max_length: max sequence length to set
    :param output_path: if provided, the model will be saved to the given path,
        otherwise, the model will be saved to a named temporary file that will
        be deleted after the program exits
    :return: if no output path, a tuple of the saved path to the model, list of
        model input names, and reference to the tempfile object will be returned
        otherwise, only the model input names will be returned
    """
    # overwrite input shapes
    model = onnx.load(path)
    initializer_input_names = set([node.name for node in model.graph.initializer])
    external_inputs = [
        inp for inp in model.graph.input if inp.name not in initializer_input_names
    ]
    input_names = []
    for external_input in external_inputs:
        # this is removed for now (will need to be accounted for when we start
        # supporting deepsparse engine
        # external_input.type.tensor_type.shape.dim[0].dim_value = batch_size
        # external_input.type.tensor_type.shape.dim[1].dim_value = max_length
        input_names.append(external_input.name)

    # Save modified model
    if output_path is None:
        tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
        onnx.save(model, tmp_file.name)

        return tmp_file.name, input_names, tmp_file
    else:
        onnx.save(model, output_path)
        return input_names


class TextGenerationInput(BaseModel):
    sequences: Union[str, List[str]] = Field(
        description="The input sequence(s) to generate "
        "text from. If a string is provided, "
        "the model will generate text from the "
        "provided sequence. If a list of strings "
        "is provided, the model will "
        "generate text from each sequence in the list.",
    )


class TextGenerationOutput(BaseModel):
    sequences: Union[str, List[str]] = Field(
        description="The input text sequence(s) appended with "
        "the generated text sequence(s). "
        "If a string was provided as input, "
        "a string will be returned. "
        "If a list of strings was provided as "
        "input, a list of strings will be returned.",
    )


@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen"],
)
class TextGenerationPipeline(TransformersPipeline):
    """
    Pipeline for text generation tasks.

    :param deterministic: if True, the pipeline will sample from
        the probability distribution computed from the logits.
        If False, the pipeline will get the next token by applying
        an argmax function to the logits.
    :param sampling_temperature: the temperature to use when sampling
        from the probability distribution computed from the logits.
        Higher values will result in more random samples.
    :param num_tokens_to_generate: the number of tokens to generate
        given the input sequence. If None, the model will generate
        tokens until the end of the sequence is reached.
    :param kwargs: kwargs to pass to the TransformersPipeline
    """

    def __init__(
        self,
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        num_tokens_to_generate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.num_tokens_to_generate = num_tokens_to_generate

        if self._batch_size != 1:
            raise ValueError(
                "For the sake of simplicity, only dynamic"
                "batch shape is supported for now. "
                "Set `batch_size` to 1 or None."
            )

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        """
        This method is used to route the input to the correct pipeline.

        :param args: args to pass to the pipeline
        :param input_schema: the input schema for the pipeline
        :param pipelines: the list of pipelines to route the input to
        :param kwargs: kwargs to pass to the pipeline
        :return: the pipeline to route the input to
        """
        raise NotImplementedError

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        Property to return the input schema for the pipeline.

        :return: the input schema for the pipeline
        """
        return TextGenerationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        Property to return the output schema for the pipeline.

        :return: the output schema for the pipeline
        """
        return TextGenerationOutput

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :return: the output schema for the pipeline
        """
        sequences = self.tokenizer.batch_decode(
            engine_outputs[0], skip_special_tokens=True
        )
        return TextGenerationOutput(sequences=sequences)

    def process_inputs(self, inputs: BaseModel) -> List[numpy.ndarray]:
        """
        Convert the input schema for the pipeline to the inputs for the engine.

        :param inputs: the input schema for the pipeline
        :return: the inputs for the engine
        """
        sequences = inputs.sequences

        if isinstance(sequences, List) and all(
            isinstance(sequence, List) and len(sequence) == 1 for sequence in sequences
        ):
            # if batch items contain only one sequence but are wrapped in lists, unwrap
            # for use as tokenizer input
            sequences = [sequence[0] for sequence in sequences]

        self.tokenizer.pad_token = self.tokenizer.eos_token

        input_tokens = self.tokenizer(
            sequences,
            return_tensors="np",
            max_length=self.sequence_length,
            padding="max_length",
        )

        input_tokens = self.add_empty_kv_cache(
            # hard-coding the shape for now
            input_tokens,
            shape=(16, self.sequence_length - 1, 64),
        )
        engine_input = self.tokens_to_engine_input(input_tokens)

        return engine_input

    def add_empty_kv_cache(
        self, input_tokens: BatchEncoding, shape: Tuple[int, ...]
    ) -> BatchEncoding:
        """
        Add empty kv_cache to the input tokens.

        :param input_tokens: the input tokens to add the kv_cache to
        :param shape: the shape of the kv_cache
        :return: the input tokens with the kv_cache added
        """
        if len(shape) == 3:
            batch_size = input_tokens["input_ids"].shape[0]
            shape = (batch_size, shape[0], shape[1], shape[2])
        if len(shape) != 4:
            raise ValueError("kv_cache shape must be 3 or 4 dimensional")

        for input_name in self.onnx_input_names:
            if input_name.startswith("past_key_values"):
                input_tokens[input_name] = numpy.zeros(shape, dtype=numpy.float32)
        return input_tokens

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], **kwargs
    ) -> numpy.ndarray:
        """
        :param engine_inputs: list of numpy inputs to
            Pipeline engine forward pass
        :return: A numpy array that contains the tokens generated by the model
        """

        # list of the meaningful tokens in the sequence
        tokens = [t for t in engine_inputs[0][0] if t != self.tokenizer.pad_token_id]

        # create an empty, initital kv_cache
        cache_keys = [
            name
            for name in self.engine._input_names
            if name.startswith("past_key_values")
        ]
        cache_arrays = [array for array in engine_inputs if numpy.all(array == 0)]
        kv_cache = dict(zip(cache_keys, cache_arrays))

        for i in range(len(tokens)):
            # pass input sequence token by token only to compute kv_cache
            # we are only processing input tokens, not adding generated ones
            _, kv_cache = self.autoregressive_pass(
                tokens=tokens[: i + 1], kv_cache=kv_cache
            )

        # establish the number of autoregressive passes to perform
        num_iterations = self.sequence_length - len(tokens)
        if self.num_tokens_to_generate:
            if self.num_tokens_to_generate > num_iterations:
                raise ValueError(
                    f"Num_tokens_to_generate ({self.num_tokens_to_generate}) "
                    f"cannot be greater than sequence_length ({self.sequence_length}) "
                    f"minus the number of tokens in the input sequence ({len(tokens)})."
                )
            num_iterations = self.num_tokens_to_generate

        # perform the remaining autoregressive passes
        for iter in range(num_iterations):
            eos_token_found = self.tokenizer.eos_token_id == tokens[-1]
            if eos_token_found:
                # fill the token list so that it has the correct sequence length
                tokens = tokens + [self.tokenizer.pad_token_id] * (
                    self.sequence_length - len(tokens)
                )
                return numpy.array([[tokens]])

            tokens, kv_cache = self.autoregressive_pass(
                tokens=tokens,
                kv_cache=kv_cache,
            )

        # fill the token list so that it has the correct sequence length
        tokens = tokens + [self.tokenizer.pad_token_id] * (
            self.sequence_length - len(tokens)
        )
        return numpy.array([[tokens]])

    def autoregressive_pass(
        self,
        tokens: List[int],
        kv_cache: Dict[str, numpy.ndarray],
    ) -> Tuple[List[int], Dict[str, numpy.ndarray]]:
        """
        Performs an autoregressive pass to generate the next token in the sequence
        and update the kv_cache with the new key/value pairs.

        1)  Set the attention mask to 1 for the tokens that are already in the sequence
            and 1 for the `new_token` - at the last position. This is because the
            `new_token`'s key/value will be added to the set of keys/values
            at the last position (before being fed to an attention block)
        2)  Set up the engine inputs
        3)  Run the engine forward pas
        4)  Preprocesses the kv cache so that it can be used as input to the next
            autoregressive pass.
        5)  Returns the new token sequence and the updated kv cache.

        :param tokens: the current token sequence
        :param kv_cache: the current kv_cache
        :return: the new token sequence and the updated kv cache
        """

        new_token = tokens[-1]

        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        attention_mask[:, : len(tokens)] = 1
        attention_mask[:, -1] = 1

        engine_inputs_dict = {
            "input_ids": numpy.array([[new_token]]),
            "attention_mask": attention_mask,
        }
        engine_inputs_dict.update(kv_cache)

        engine_inputs = [engine_inputs_dict[name] for name in self.engine._input_names]

        new_logits, *new_kvs = self.engine(engine_inputs)

        # rename the output names to match the names expected
        # in the next autoregressive pass
        kv_output_names = [
            name.replace("present", "past_key_values")
            for name in self.engine._output_names
            if name.startswith("present")
        ]
        kv_cache = dict(zip(kv_output_names, new_kvs))
        for k, v in kv_cache.items():
            v[:, :, len(tokens) - 1] = v[:, :, -1]
            kv_cache[k] = numpy.ascontiguousarray(v[:, :, :-1])

        # Obtain the next token from the logits
        new_token = TextGenerationPipeline.sample_new_token(
            logits=new_logits[0, -1, :],
            deterministic=self.deterministic,
            temperature=self.sampling_temperature,
        )
        tokens.append(new_token)

        return tokens, kv_cache

    @staticmethod
    def sample_new_token(
        logits: numpy.ndarray, deterministic: bool, temperature: float
    ) -> int:
        """
        Samples a token from the logits using the sampling temperature.

        :param logits: the logits from the model
        :param deterministic: whether to sample from the softmax or take the argmax
        :param temperature: the sampling temperature

        :return: the sampled token
        """
        if deterministic:
            return numpy.argmax(logits)
        else:
            logits /= temperature
            probs = softmax(logits)
            return numpy.random.choice(len(probs), p=probs)

    def setup_onnx_file_path(self) -> str:
        # `setup_onnx_file_path` function rewritten to take into account modified
        # `overwrite_transformer_onnx_model_inputs` function

        """
        Parses ONNX, tokenizer, and config file paths from the given `model_path`.
        Supports sparsezoo stubs

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path, config_path, tokenizer_path = get_onnx_path_and_configs(
            self.model_path, require_configs=True
        )

        self.config = AutoConfig.from_pretrained(
            config_path, finetuning_task=self.task if hasattr(self, "task") else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            model_max_length=self.sequence_length,
        )
        self.config_path = os.path.join(config_path, "config.json")
        self.tokenizer_config_path = os.path.join(tokenizer_path, "tokenizer.json")

        # overwrite onnx graph to given required input shape
        (
            onnx_path,
            self.onnx_input_names,
            self._temp_model_directory,
        ) = overwrite_transformer_onnx_model_inputs(
            onnx_path, max_length=self.sequence_length
        )

        return onnx_path
