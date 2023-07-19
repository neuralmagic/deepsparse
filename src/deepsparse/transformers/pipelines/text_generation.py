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

import logging
import warnings
from typing import List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.cpu import cpu_avx512_compatible
from deepsparse.pipeline import DEEPSPARSE_ENGINE
from deepsparse.transformers.engines import NLDecoderEngine
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.utils.helpers import pad_to_fixed_length


_LOGGER = logging.getLogger(__name__)

__all__ = ["TextGenerationPipeline"]


class TextGenerationInput(BaseModel):
    sequences: Union[str, List[str]] = Field(
        description="The input sequences to generate the text from.",
    )
    return_logits: bool = Field(
        default=False,
        description="A flag that indicates whether to return "
        "the logits for the input text sequence and the "
        "generated text sequence. ",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="A user may set a string identifier "
        "for the kv cache session. If None, "
        "and the model is using kv cache, it "
        "will be set to a random uuid.",
    )
    fixed_sequences_length: bool = Field(
        default=False,
        description="A flag that indicates whether to modify "
        "(pad or truncate) each input text sequence, so that "
        "its tokenized length is equal to `sequence_length` "
        "of tokens. Useful, when a batch of predictions needs "
        "to have consistent length so one "
        "can compute metric in a batched fashion. ",
    )


class TextGenerationOutput(BaseModel):
    sequences: Union[str, List[str]] = Field(
        description="The generated text sequences.",
    )
    logits: Optional[numpy.ndarray] = Field(
        default=None,
        description="The logits for the generated text sequence."
        "The logits have dimensions "
        "[batch_size, sequence_length, vocab_size]",
    )
    session_id: Optional[str] = Field(
        default=None, description="A string identifier for the kv cache session."
    )

    class Config:
        arbitrary_types_allowed = True


@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen", "opt", "bloom"],
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
        Higher values will result in more random samples. Should
        be greater than 0.0.
    :param max_generated_tokens: the maximum number of tokens to generate
        given the input sequence. If None, the model will generate
        tokens until the end of the sequence is reached.
        Otherwise, it will generate up to the maximum number of tokens or end of
        sequence is reached.
    :param prompt_processing_sequence_length: For large prompts, the prompt is
        processed in chunks of this length. This is to maximize the inference
        speed. By default, this is set to 128.
    :param force_max_tokens: if True, the pipeline will generate the maximum number
        of tokens supplied even if the stop token is reached.
    :param use_deepsparse_cache: if True, the pipeline will use the deepsparse kv cache
        for caching the model outputs.
    :param kwargs: kwargs to pass to the TransformersPipeline
    """

    def __init__(
        self,
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        max_generated_tokens: Optional[int] = 1024,
        # TODO: Set this to 64 once we modify the OPT injection logic
        prompt_processing_sequence_length: int = 128,
        force_max_tokens: bool = False,
        use_deepsparse_cache: bool = True,
        **kwargs,
    ):
        if not cpu_avx512_compatible() and kwargs["engine_type"] == DEEPSPARSE_ENGINE:
            warnings.warn(
                "AVX512 support not detected, disabling internal management "
                "of KV cache which may affect performance. To enable full "
                "performance, deploy on an AVX512-compatible system."
            )
            use_deepsparse_cache = False

        if use_deepsparse_cache:
            if kwargs["engine_type"] != DEEPSPARSE_ENGINE:
                raise ValueError(
                    "`use_deepsparse_cache` is set to True "
                    "but the chosen `engine_type` "
                    f"is {kwargs['engine_type']}. "
                    f"Make sure to set `engine_type` to {DEEPSPARSE_ENGINE}"
                )

        super().__init__(
            **kwargs, _delay_engine_initialize=True, _delay_overwriting_inputs=True
        )

        if self.engine_type == DEEPSPARSE_ENGINE:
            _LOGGER.warning(
                "The support for deepsparse engine is limited "
                f"for {self.__class__.__name__}. "
                "The multi-token engine will not be "
                "used for prompt processing."
            )

        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.max_generated_tokens = max_generated_tokens
        self.prompt_processing_sequence_length = prompt_processing_sequence_length
        self.force_max_tokens = force_max_tokens

        # override tokenizer to pad to left
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.engine = None

        self.multitoken_engine = NLDecoderEngine(
            onnx_file_path=self.onnx_file_path,
            engine_type=self.engine_type,
            engine_args=self.engine_args,
            engine_context=self.context,
            sampling_temperature=self.sampling_temperature,
            deterministic=self.deterministic,
            sequence_length=self.sequence_length,
            input_ids_length=prompt_processing_sequence_length,
            tokenizer=self.tokenizer,
            use_deepsparse_cache=use_deepsparse_cache,
        )

        if self.multitoken_engine.kv_cache_enabled:
            # unless kv cache is enabled, we don't
            # need to initialize the single token engine
            self.engine = NLDecoderEngine(
                onnx_file_path=self.onnx_file_path,
                engine_type=self.engine_type,
                engine_args=self.engine_args,
                engine_context=self.context,
                sampling_temperature=self.sampling_temperature,
                deterministic=self.deterministic,
                sequence_length=self.sequence_length,
                input_ids_length=1,
                tokenizer=self.tokenizer,
                use_deepsparse_cache=use_deepsparse_cache,
            )
        if (
            not self.multitoken_engine.kv_cache_enabled
            and self.max_generated_tokens > 1
        ):
            raise ValueError(
                "The model used for inference does not support kv cache. It is "
                "assumed that it maps from the token sequence to predicted logits."
                "Set `max_generated_tokens` to 1 to support that scenario."
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

        if inputs.fixed_sequences_length:
            # to enforce a fixed sequence length, we need to
            # truncate the input to the maximum sequence length
            # or/and pad it to the maximum sequence length
            truncate, padding = True, "max_length"
        else:
            # otherwise, we do not need to truncate the input
            # and we shall can pad it to the longest sequence
            # in the batch (so that the engine can process multiple inputs
            # at once)
            truncate, padding = False, "longest"

        input_tokens = self.tokenizer(
            inputs.sequences,
            return_tensors="np",
            max_length=self.sequence_length,
            padding=padding,
            truncation=truncate,
        )

        attention_mask = input_tokens["attention_mask"]

        # TODO: Positions input is not required by BLOOM
        # let's make it optional in the future
        positions = attention_mask.cumsum(1) * attention_mask
        positions -= 1  # assert that positions start at 0
        positions_input = dict(positions=positions)

        input_tokens = {**input_tokens, **positions_input}
        onnx_input_names = self.multitoken_engine.onnx_input_names_no_cache
        engine_input = self.tokens_to_engine_input(input_tokens, onnx_input_names)

        if inputs.session_id is not None:
            # if session_id is provided, we need to set it in engines
            self.engine.session_id = inputs.session_id
            self.multitoken_engine.session_id = inputs.session_id

        postprocessing_kwargs = dict(return_logits=inputs.return_logits)
        return engine_input, postprocessing_kwargs

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> TextGenerationOutput:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :return: the output schema for the pipeline
        """
        generated_tokens, generated_logits = engine_outputs
        sequences = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        logits = generated_logits if kwargs.get("return_logits") else None

        return TextGenerationOutput(sequences=sequences, logits=logits)

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], **kwargs
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Run the forward pass on the engine.

        :param engine_inputs: list of numpy inputs to
            Pipeline engine forward pass
        :return: A tuple of numpy array that contains the
            sequence of generated tokens and a sequence
            of logits for each generated token
        """
        if not self.multitoken_engine.kv_cache_enabled:
            tokens, prompt_logits = self.multitoken_engine(engine_inputs)
            return numpy.array([tokens]), prompt_logits

        else:
            # run the prompt through
            tokens, prompt_logits = self.prompt_inference(engine_inputs)

        # create the generated output
        max_tokens = (
            self.max_generated_tokens
            if self.max_generated_tokens and self.max_generated_tokens > 0
            else 100 * self.sequence_length
        )  # set safety for absolute max generation

        generated_tokens = [tokens[-1]]
        generated_logits = prompt_logits

        while len(generated_tokens) < max_tokens:
            (
                token,
                logits,
            ) = self.autoregressive_inference(tokens)
            tokens.append(token)
            generated_tokens.append(token)
            generated_logits.append(logits)

            if token == self.tokenizer.eos_token_id and not self.force_max_tokens:
                break

        return numpy.array([generated_tokens]), numpy.concatenate(
            generated_logits, axis=1
        )

    def prompt_inference(
        self, engine_inputs: List[numpy.ndarray]
    ) -> Tuple[List[int], List[numpy.ndarray]]:
        """
        An inference run that processes the prompt through the
        model to generate the new token and logits

        :param engine_inputs: the prompt (context) represented by a
            list of numpy inputs to the engine
        :return: A tuple of:
            - The list of prompt tokens plus the new, generated token
            - The logits generated from the prompt (with dimensions
            ['batch_size', 'num_tokens', 'vocab_size'])
        """
        # get tokens by attention mask
        tokens = engine_inputs[0][engine_inputs[1].nonzero()].tolist()

        prompt_logits = []
        new_token = None
        num_tokens_processed = 0

        # clean the state of engines' cache
        # in the future, this will be paired with the session ids
        # to refrain from resetting if session id is being passed
        self._reset_engines_cache()

        # TODO: Multiple passes through the multitoken
        # engine once the OPT injection is fixed
        if (
            len(tokens) > self.prompt_processing_sequence_length
            and self.engine_type != DEEPSPARSE_ENGINE
        ):
            # trim the input to the prompt size
            engine_inputs = [
                input[:, : self.prompt_processing_sequence_length]
                for input in engine_inputs
            ]
            new_token, new_logits = self.multitoken_engine(engine_inputs)
            num_tokens_processed = self.prompt_processing_sequence_length
            prompt_logits.append(new_logits)

        if num_tokens_processed:
            # transfer the cache state from the multi-token engine to the main engine
            self.engine.transfer_cache_state(cache=self.multitoken_engine.kv_cache)

        # prompt size is small, run autoregressive inference to populate kv cache
        run_tokens = [] if num_tokens_processed == 0 else tokens[:num_tokens_processed]

        for token in tokens[num_tokens_processed:]:
            run_tokens.append(token)
            new_token, new_logits = self.autoregressive_inference(
                run_tokens, shift_positions_by_one=not bool(num_tokens_processed)
            )
            prompt_logits.append(new_logits)

        tokens.append(new_token)

        return tokens, prompt_logits

    def autoregressive_inference(
        self,
        tokens: List[int],
        shift_positions_by_one: bool = False,
    ) -> Tuple[int, numpy.ndarray]:
        """
        An inference run that processes the last token to generate
        a new token and new logits.

        :param tokens: The current context (prompt + generated tokens so far)
        :param shift_positions_by_one: Whether to shift the positions
            by one. Used if we are processing the prompt from the scratch
            (i.e. not using the multitoken engine)
        :return: The new, generated token and the logits for the new token
            (with dimensions ['batch_size', 'num_tokens', 'vocab_size'])
        """
        new_token = tokens[-1]
        # padding is added to left, so attention mask is 1s from the
        # right up to the number of total tokens (prompt + generated)
        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        num_tokens_processed = min(len(tokens), self.sequence_length)  # cap by seq len
        attention_mask[:, -num_tokens_processed:] = 1
        positions = numpy.array([[len(tokens)]], dtype=numpy.int64)
        if shift_positions_by_one:
            positions -= 1
        input_ids = numpy.array([[new_token]])
        engine_inputs = [input_ids, attention_mask, positions]

        generated_token, generated_logits = self.engine(engine_inputs)

        return generated_token, generated_logits

    @property
    def has_cache(self) -> bool:
        """
        Returns whether the ran model has kv cache or not

        :return: True if the model has kv cache, False otherwise
        """
        return self.multitoken_engine.kv_cache_enabled

    @staticmethod
    def join_engine_outputs(
        batch_outputs: List[List[numpy.ndarray]],
    ) -> List[numpy.ndarray]:
        """
        Takes a list of outputs (batches) from the engine
        and joins them into a single output. Asserts that
        the dimensions of the outputs are the same, so that
        they can be concatenated.

        :param batch_outputs: A list of outputs from the engine
        :return: A list of joined outputs
        """
        tokens, logits = zip(*batch_outputs)
        tokens = numpy.concatenate(tokens, axis=0)
        # find the longest sequence in the batch of logits
        max_len = max([logits.shape[1] for logits in logits])
        # pad all logits to the same length
        logits = [
            pad_to_fixed_length(array=single_logits, max_len=max_len, axis=1)
            for single_logits in logits
        ]
        logits = numpy.concatenate(logits, axis=0)
        return [tokens, logits]

    def _reset_engines_cache(self):
        self.engine.reset_kv_cache()
        self.multitoken_engine.reset_kv_cache()
