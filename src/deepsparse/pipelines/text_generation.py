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
from typing import Dict, List, Optional, Tuple, Type

import numpy
import onnx
from onnx import ValueInfoProto
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer

from deepsparse import Pipeline
from deepsparse.transformers.helpers import (
    get_onnx_path_and_configs,
    overwrite_transformer_onnx_model_inputs,
)
from deepsparse.transformers.pipelines import TransformersPipeline


# TODO: to be deprecated after Sage's changes, we will only need a single model
_MODEL_DIR_ONNX_MULTI_TOKEN_NAME = "decoder_model.onnx"

__all__ = ["TextGenerationPipeline"]


def softmax(x: numpy.ndarray) -> numpy.ndarray:
    """
    Compute softmax values for x
    :param x: input array
    :return: softmax values
    """
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)


class TextGenerationInput(BaseModel):
    sequence: str = Field(
        description="The input sequence to generate the text from.",
    )


class TextGenerationOutput(BaseModel):
    sequence: str = Field(
        description="The generated text sequence.",
    )


@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen", "opt"],
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
    :param max_generated_tokens: the maximum number of tokens to generate
        given the input sequence. If None, the model will generate
        tokens until the end of the sequence is reached.
        Otherwise it will generate up to the maximum number of tokens or end of
        sequence is reached.
    :param kwargs: kwargs to pass to the TransformersPipeline
    """

    def __init__(
        self,
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        max_generated_tokens: Optional[int] = 1024,
        prompt_batch_threshold: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs, _delay_engine_initialize=True)

        if self._batch_size != 1:
            raise ValueError("Only batch size 1 is supported for generation pipelines")

        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.max_generated_tokens = max_generated_tokens
        self.prompt_batch_threshold = prompt_batch_threshold
        # when we are done with the static inference in ORT,
        # set support_kv_cache = True
        self.engine = Pipeline.create_engine(
            self.onnx_file_path,
            self.engine_type,
            self.engine_args,
            self.context,
        )

        self.onnx_multitoken_path = self.setup_onnx_file_path(multitoken=True)
        # initialize the auxiliary multitoken engine
        self.multitoken_engine = Pipeline.create_engine(
            self.onnx_multitoken_path, self.engine_type, self.engine_args, self.context
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
        raise ValueError("Bucketing is not supported for generation pipelines")

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

    def process_inputs(self, inputs: TextGenerationInput) -> List[numpy.ndarray]:
        """
        Convert the input schema for the pipeline to the inputs for the engine.

        :param inputs: the input schema for the pipeline
        :return: the inputs for the engine
        """

        self.tokenizer.pad_token = self.tokenizer.eos_token

        input_tokens = self.tokenizer(
            inputs.sequence,
            return_tensors="np",
            max_length=self.sequence_length,
            padding="max_length",
        )

        kv_cache = self._initialize_kv_cache()

        attention_mask = input_tokens["attention_mask"]
        positions = attention_mask.cumsum(1) * attention_mask
        positions -= 1  # zero index
        positions_input = dict(positions=positions)

        input_tokens = {**input_tokens, **kv_cache, **positions_input}
        engine_input = self.tokens_to_engine_input(input_tokens)

        return engine_input

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> TextGenerationOutput:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :return: the output schema for the pipeline
        """
        sequence = self.tokenizer.decode(engine_outputs[0][0], skip_special_tokens=True)
        return TextGenerationOutput(sequence=sequence)

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], **kwargs
    ) -> numpy.ndarray:
        """
        Run the forward pass on the engine.

        :param engine_inputs: list of numpy inputs to
            Pipeline engine forward pass
        :return: A numpy array that contains the tokens generated by the model
        """
        # run the prompt through
        tokens, kv_cache = self.prompt_inference(engine_inputs)
        num_prompt_tokens = len(tokens) - 1

        # create the generated output
        # TODO: Apply sliding window logic
        max_tokens = (
            self.max_generated_tokens
            if self.max_generated_tokens and self.max_generated_tokens > 0
            else 100 * self.sequence_length
        )  # set safety for absolute max generation
        generated = [tokens[-1]]

        while len(generated) < max_tokens:
            gen_token, kv_cache = self.autoregressive_inference(
                tokens, kv_cache, num_prompt_tokens
            )
            tokens.append(gen_token)
            generated.append(gen_token)

            if gen_token == self.tokenizer.eos_token_id:
                break

        return numpy.array([[generated]])

    def prompt_inference(
        self, engine_inputs: List[numpy.ndarray]
    ) -> Tuple[List[int], Dict[str, numpy.ndarray]]:
        """
        An inference run that processes the prompt through the
        model to generate the new token and populate the kv cache.

        :param engine_inputs: the prompt (context) represented by a
            list of numpy inputs to the engine
        :return:
            - the list of prompt tokens plus the new, generated token
            - the kv cache that was populated during the inference
        """
        tokens = [t for t in engine_inputs[0][0] if t != self.tokenizer.pad_token_id]
        new_token = None

        if len(tokens) / float(self.sequence_length) < self.prompt_batch_threshold:
            # prompt size is small, run autoregressive inference to populate kv cache
            run_tokens = []
            kv_cache = {}
            for token in tokens:
                run_tokens.append(token)
                new_token, kv_cache = self.autoregressive_inference(
                    run_tokens, kv_cache
                )
        else:
            # larger prompt size, run through multi-token engine in single pass
            logits, *cache_values = self.multitoken_engine(engine_inputs)
            kv_cache = self._assemble_kv_cache(
                cache_values, tokens, len(tokens) - 1
            )
            new_token = self.generate_token(logits[0, : len(tokens) + 1])

        tokens.append(new_token)

        return tokens, kv_cache

    def autoregressive_inference(
        self,
        tokens: List[int],
        kv_cache: Dict[str, numpy.ndarray],
        num_prompt_tokens: int,
    ) -> Tuple[int, Dict[str, numpy.ndarray]]:
        """
        An inference run that processes the last token and the kv cache to
        generate a new token and update the kv cache.

        :param tokens: The current context (prompt + generated tokens so far)
        :param kv_cache: The key-value cache from the previous inference run
        :param num_prompt_tokens: number of tokens in the initial prompt
        :return:
            - the list of prompt tokens plus the new, generated token
            - the kv cache that was populated during the inference
        """
        new_token = tokens[-1]
        num_generated_tokens = len(tokens) - num_prompt_tokens

        # due to right hand concatenation, attention mask is:
        # 1s for length of prompt + 1s starting from RHS for generated tokens + 1
        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        attention_mask[:, :num_prompt_tokens + 1] = 1  # +1 because
        # OPT adds an initial pad
        # fill in generated tokens from RHS
        attention_mask[:, -(min(num_generated_tokens, self.sequence_length)):] = 1

        # the position of the token is the number of tokens - 1 (zero indexed)
        positions = numpy.array([[len(tokens)]], dtype=numpy.int64)
        engine_inputs = {
            "input_ids": numpy.array([[new_token]]),
            "attention_mask": attention_mask,
            "positions": positions,
        }

        engine_inputs.update(kv_cache)
        engine_inputs = [engine_inputs[name] for name in self.engine.input_names]

        new_logits, *cache_values = self.engine(engine_inputs)
        kv_cache = self._assemble_kv_cache(
            cache_values, tokens, num_prompt_tokens
        )

        # Obtain the next token from the logits
        generated_token = self.generate_token(new_logits[0, 0, :])

        return generated_token, kv_cache

    def generate_token(self, logits: numpy.ndarray) -> int:
        """
        Samples a token from the logits using the sampling temperature.

        :param logits: the logits from the model

        :return: the sampled token
        """
        if self.deterministic:
            return numpy.argmax(logits)

        logits /= self.sampling_temperature

        probs = softmax(logits)

        return numpy.random.choice(len(probs), p=probs)

    def setup_onnx_file_path(self, multitoken: bool = False):
        """
        Parses ONNX, tokenizer, and config file paths from the given `model_path`.
        Supports sparsezoo stubs

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path, config_path, tokenizer_path = get_onnx_path_and_configs(
            self.model_path,
            require_configs=True,
            model_dir_onnx_name="model_kv_cache.onnx",
        )

        self.config = AutoConfig.from_pretrained(config_path)

        # So far hardcoding the tokenizer, to be figured out later
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-350m",
            model_max_length=self.sequence_length,
        )
        self.config_path = os.path.join(config_path, "config.json")

        # overwrite onnx graph to given required input shape
        (
            onnx_path,
            self.onnx_input_names,
            self._temp_model_directory,
        ) = overwrite_transformer_onnx_model_inputs(
            onnx_path,
            max_length=self.sequence_length,
            custom_input_overwrite_func=self.overwrite_onnx_model_inputs,
            custom_input_overwrite_func_kwargs=dict(
                multitoken=multitoken,
            ),
        )

        model = onnx.load_model(onnx_path, load_external_data=False)
        self.external_outputs = [out for out in model.graph.output]

        return onnx_path

    def _initialize_kv_cache(self):
        # initialize empty kv cache
        empty_kv_cache_tensor = numpy.zeros(
            # hard coded for now, we can fetch it automatically
            # from engine output shapes, but because it overwrites
            # 16 with 1, not feasible for now
            (
                16,  # num heads
                0,
                64,  # hidden dims
            ),
            dtype=numpy.float32,
        )  # hidden size

        cache_keys = [
            output.name.replace("present", "past_key_values")
            for output in self.external_outputs
            if output.name.startswith("present")
        ]
        return {key: empty_kv_cache_tensor for key in cache_keys}

    def _assemble_kv_cache(
        self,
        cache_values: List[numpy.ndarray],
        tokens: List[int],
        num_prompt_tokens: int,
    ) -> Dict[str, numpy.ndarray]:
        # first, trim the output cache (seq_len) to input cache shape (seq_len - 1)
        for idx, cache_value in enumerate(cache_values):
            if len(tokens) > self.sequence_length - 1:
                # all values in cache are from non-pad tokens, pop from front
                cache_values[idx] = cache_value[:, 1:, :]
            else:
                # remove the cache key/value immediately after the prompt since this
                # is where the last padded value will go
                idxs_to_keep = [
                    idx
                    for idx in range(self.sequence_length)
                    if idx != num_prompt_tokens + 2
                    # adding +1 is OPT specific since they always add an extra token
                ]
                cache_values[idx] = cache_value[:, idxs_to_keep, :]

        # rename the output names to match the names expected
        # in the next autoregressive pass
        cache_keys = [
            name.replace("present", "past_key_values")
            for name in self.engine.output_names
            if name.startswith("present")
        ]
        kv_cache = dict(zip(cache_keys, cache_values))

        return kv_cache

    @staticmethod
    def overwrite_onnx_model_inputs(
        external_inputs: List[ValueInfoProto],
        batch_size: int,
        sequence_length: int,
        multitoken: bool,
    ) -> List[str]:
        """
        Overwrite the input shape of the onnx model.

        :param external_inputs: the external inputs of the onnx model
        :param batch_size: the batch size of the input
        :param max_length: the max length of the input
        :param multitoken: true if model is to be run with seq len > 1
        :return: the input names of the onnx model
        """
        input_names = []
        for external_input in external_inputs:
            if external_input.name in ["input_ids", "positions"]:
                external_input.type.tensor_type.shape.dim[0].dim_value = batch_size
                external_input.type.tensor_type.shape.dim[1].dim_value = (
                    sequence_length if multitoken else 1
                )
            elif external_input.name == "attention_mask":
                external_input.type.tensor_type.shape.dim[0].dim_value = batch_size
                # even in single token cached runs, full attention mask
                # will be provided
                external_input.type.tensor_type.shape.dim[1].dim_value = sequence_length
            elif external_input.name.startswith("past_key_values"):
                external_input.type.tensor_type.shape.dim[0].dim_value = 16  # n_heads
                # no cache for multitoken runs, otherwise max cache len is max len - 1
                external_input.type.tensor_type.shape.dim[1].dim_value = (
                    0 if multitoken else sequence_length - 1
                )
                external_input.type.tensor_type.shape.dim[
                    2
                ].dim_value = 64  # hidden dims
            else:
                raise ValueError(f"Unexpected input name: {external_input.name}")

            input_names.append(external_input.name)
        return input_names
