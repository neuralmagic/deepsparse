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

from typing import Dict, List, Optional, Tuple, Type, Union

import numpy
import onnx
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline
from scipy.special import softmax


_MODEL_DIR_ONNX_MULTI_TOKEN_NAME = "decoder_model.onnx"
_MODEL_DIR_ONNX_NAME = "model.onnx"

__all__ = ["TextGenerationPipeline"]


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
        if self._batch_size != 1:
            raise ValueError("Only batch size 1 is supported for generation pipelines")

        super().__init__(**kwargs, _delay_engine_initialize=True)
        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.max_generated_tokens = max_generated_tokens
        self.prompt_batch_threshold = prompt_batch_threshold

        # setup the single token engine -- used to continually generate tokens
        self._adapt_onnx_file_sequence_len(sequence_length=1)
        self._initialize_engine()

        # setup the multitoken engine -- used for large inputs to generate kv cache
        self._adapt_onnx_file_sequence_len(sequence_length=self.sequence_length)
        self.multitoken_engine = Pipeline.create_engine(
            self.onnx_file_path, self.engine_type, self.engine_args, self.context
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

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> TextGenerationOutput:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :return: the output schema for the pipeline
        """
        sequences = self.tokenizer.batch_decode(
            engine_outputs[0], skip_special_tokens=True
        )
        return TextGenerationOutput(sequences=sequences)

    def process_inputs(self, inputs: TextGenerationInput) -> List[numpy.ndarray]:
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

        engine_input = self.tokens_to_engine_input(
            input_tokens, onnx_input_names=self.onnx_input_names
        )

        return engine_input

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], **kwargs
    ) -> numpy.ndarray:
        """
        :param engine_inputs: list of numpy inputs to
            Pipeline engine forward pass
        :return: A numpy array that contains the tokens generated by the model
        """
        # run the prompt through
        tokens, kv_cache = self.prompt_inference(engine_inputs)

        # create the generated output
        max_tokens = (
            self.max_generated_tokens
            if self.max_generated_tokens and self.max_generated_tokens > 0
            else 100 * self.sequence_length
        )  # set safety for absolute max generation
        generated = []

        while len(generated) < max_tokens:
            gen_token, kv_cache = self.autoregressive_inference(tokens, kv_cache)
            tokens.append(gen_token)
            generated.append(gen_token)

            if gen_token == self.tokenizer.eos_token_id:
                break

        return numpy.array([[generated]])

    def autoregressive_inference(
        self, tokens: List[int], kv_cache: Dict[str, numpy.ndarray]
    ) -> Tuple[int, Dict[str, numpy.ndarray]]:
        """

        :param tokens:
        :param kv_cache:
        :return:
        """
        new_token = tokens[-1]

        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        attention_mask[:, : len(tokens)] = 1
        attention_mask[:, -1] = 1

        engine_inputs = {
            "input_ids": numpy.array([[new_token]]),
            "attention_mask": attention_mask,
        }
        engine_inputs.update(kv_cache)
        engine_inputs = [engine_inputs[name] for name in self.engine._input_names]

        new_logits, *cache_values = self.engine(engine_inputs)
        kv_cache = self._assemble_kv_cache(cache_values, tokens)

        # Obtain the next token from the logits
        generated_token = self.generate_token(new_logits[0, -1, :])

        return generated_token, kv_cache

    def prompt_inference(
        self, engine_inputs: List[numpy.ndarray]
    ) -> Tuple[List[int], Dict[str, numpy.ndarray]]:
        """

        :param engine_inputs:
        :return:
        """
        tokens = [t for t in engine_inputs[0][0] if t != self.tokenizer.pad_token_id]
        new_token = None

        if len(tokens) / float(self.sequence_length) < self.prompt_batch_threshold:
            # prompt size is small, run autoregressive inference to populate kv cache
            run_tokens = []
            kv_cache = {}
            for token in tokens:
                run_tokens.append(token)
                new_token, kv_cache = self.autoregressive_inference(run_tokens, kv_cache)
        else:
            # larger prompt size, run through multitoken engine in single pass
            logits, *cache_values = self.multitoken_engine(engine_inputs)
            kv_cache = self._assemble_kv_cache(cache_values, tokens)
            new_token = self.generate_token(logits[0, len(tokens) - 1])

        tokens.append(new_token)

        return tokens, kv_cache

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

    def _assemble_kv_cache(
        self, cache_values: List[numpy.ndarray], tokens: List[int]
    ) -> Dict[str, numpy.ndarray]:
        # rename the output names to match the names expected
        # in the next autoregressive pass
        cache_keys = [
            name.replace("present", "past_key_values")
            for name in self.engine._output_names
            if name.startswith("present")
        ]
        kv_cache = dict(zip(cache_keys, cache_values))
        for key, val in kv_cache.items():
            val[:, :, len(tokens) - 1] = val[:, :, -1]
            kv_cache[key] = numpy.ascontiguousarray(val[:, :, :-1])

        return kv_cache

    def _adapt_onnx_file_sequence_len(self, sequence_length: int):
        model = onnx.load(self.onnx_file_path)
        initializer_input_names = set([node.name for node in model.graph.initializer])
        external_inputs = [
            inp for inp in model.graph.input if inp.name not in initializer_input_names
        ]
        input_names = []
        for external_input in external_inputs:
            # this is removed for now (will need to be accounted for when we start
            # supporting deepsparse engine
            external_input.type.tensor_type.shape.dim[0].dim_value = 1
            external_input.type.tensor_type.shape.dim[1].dim_value = sequence_length
            input_names.append(external_input.name)

        onnx.save(model, self.onnx_file_path)
