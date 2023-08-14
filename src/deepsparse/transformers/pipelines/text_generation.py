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
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import numpy
import onnx
from pydantic import BaseModel, Field
from transformers import TextStreamer

from deepsparse import Pipeline
from deepsparse.pipeline import DEEPSPARSE_ENGINE
from deepsparse.transformers.engines import NLDecoderEngine
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.utils.helpers import (
    create_causal_mask,
    pad_to_fixed_length,
)
from deepsparse.utils.onnx import default_cached_outputs


_LOGGER = logging.getLogger(__name__)

__all__ = ["TextGenerationPipeline"]


@dataclass(frozen=True)
class _TextGenerationTimings:
    PROMPT_PREFILL: str = "engine_prompt_prefill"
    PROMPT_PREFILL_SINGLE: str = "engine_prompt_prefill_single"
    TOKEN_GENERATION: str = "engine_token_generation"
    TOKEN_GENERATION_SINGLE: str = "engine_token_generation_single"


class TextGenerationInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

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
    streamer: Optional[TextStreamer] = Field(
        default=None,
        description="Streamer object that will be used to stream the "
        "generated sequences. Generated tokens are passed through "
        "`streamer.put(token_ids)` and the streamer is responsible "
        "for any further processing.",
    )


class TextGenerationOutput(BaseModel):
    sequences: Union[str, List[str]] = Field(
        description="The generated text sequences.",
    )
    logits: Optional[Any] = Field(  # numpy array, set to Any for FastAPI compatibility
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
        prompt_processing_sequence_length: int = 64,
        force_max_tokens: bool = False,
        use_deepsparse_cache: bool = True,
        **kwargs,
    ):
        kwargs_engine_type = kwargs.get("engine_type", DEEPSPARSE_ENGINE)

        if use_deepsparse_cache:
            if kwargs_engine_type != DEEPSPARSE_ENGINE:
                _LOGGER.warning(
                    "`use_deepsparse_cache` is set to True "
                    "but the chosen `engine_type` "
                    f"is {kwargs_engine_type}. "
                    f"The optimized kv cache management is disabled."
                )
                use_deepsparse_cache = False

        super().__init__(
            **kwargs, _delay_engine_initialize=True, _delay_overwriting_inputs=True
        )
        self.enable_multitoken_prefill = self.causal_mask_input_present(
            model_path=self.onnx_file_path
        )
        self.cache_support_enabled = self.is_cache_support_enabled()

        if self.engine_type == DEEPSPARSE_ENGINE:
            if "WAND_OPT_FLAGS" not in os.environ:
                os.environ["WAND_OPT_FLAGS"] = "default,~pyramids"

        if not self.cache_support_enabled and max_generated_tokens > 1:
            raise ValueError(
                "The model used for inference does not support kv cache. It is "
                "assumed that it maps from the token sequence to predicted logits."
                "Set `max_generated_tokens` to 1 to support that scenario."
            )

        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.max_generated_tokens = max_generated_tokens
        self.prompt_processing_sequence_length = prompt_processing_sequence_length
        self.force_max_tokens = force_max_tokens
        self.use_deepsparse_cache = use_deepsparse_cache

        # override tokenizer to pad to left
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.engine, self.multitoken_engine = self.initialize_engines()

    def initialize_engines(
        self,
    ) -> Tuple[Optional[NLDecoderEngine], Optional[NLDecoderEngine]]:
        """
        Inititalizes a pair of engines for the pipeline.
        The first engine (`engine`) is used for processing the tokens token-by-token
        (in the autoregressive fashion).
        The second engine (`multitoken_engine`) is used for processing the tokens
        in a single pass (in the multitoken fashion).

        There are several cases of how the engines are initialized:
        - if the model does not support kv cache, then only the
            `multitoken_engine` is initialized. The `engine` is set to None.
        - if the model supports kv cache but does not support
            multitoken prefill scenario (i.e. self.enable_multitoken_prefill = False),
            then only the `engine` is initialized. The `multitoken_engine`
            is set to None.

        :return: a pair of engines (`engine`, `multitoken_engine`)
            Note: that depending on the scenario one of the engines may be None
        """

        engine, multitoken_engine = None, None

        if self.cache_support_enabled:
            # emit the appropriate user message depending whether we are
            # instantiation the multitoken engine or not
            if not self.enable_multitoken_prefill:
                warnings.warn(
                    "This ONNX graph does not support processing the prompt in "
                    "with processing length > 1. Creation of an auxiliary engine for "
                    "processing the prompt at a larger processing length is disabled. "
                    "The prompt will be processed in with processing length 1."
                )
            else:
                _LOGGER.info(
                    "Compiling an auxiliary engine to process a prompt with a "
                    "larger processing length. This improves performance, but "
                    "may result in additional memory consumption."
                )

        if (
            self.cache_support_enabled and self.enable_multitoken_prefill
        ) or not self.cache_support_enabled:

            multitoken_engine = NLDecoderEngine(
                onnx_file_path=self.onnx_file_path,
                engine_type=self.engine_type,
                engine_args=self.engine_args,
                engine_context=self.context,
                sampling_temperature=self.sampling_temperature,
                deterministic=self.deterministic,
                sequence_length=self.sequence_length,
                input_ids_length=self.prompt_processing_sequence_length,
                tokenizer=self.tokenizer,
                use_deepsparse_cache=self.use_deepsparse_cache,
            )

        if self.cache_support_enabled:

            engine = NLDecoderEngine(
                onnx_file_path=self.onnx_file_path,
                engine_type=self.engine_type,
                engine_args=self.engine_args,
                engine_context=self.context,
                sampling_temperature=self.sampling_temperature,
                deterministic=self.deterministic,
                sequence_length=self.sequence_length,
                input_ids_length=1,
                tokenizer=self.tokenizer,
                use_deepsparse_cache=self.use_deepsparse_cache,
            )

        assert (engine is not None) or (
            multitoken_engine is not None
        ), "At least one of the engines must be initialized for the pipeline!"
        return engine, multitoken_engine

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

        positions = attention_mask.cumsum(1) * attention_mask
        positions -= 1  # assert that positions start at 0

        causal_mask = create_causal_mask(
            input_tokens["input_ids"], input_tokens["attention_mask"]
        )

        input_tokens = dict(
            **input_tokens, positions=positions, causal_mask=causal_mask
        )
        onnx_input_names = (
            self.multitoken_engine.onnx_input_names_no_cache
            if self.multitoken_engine
            else self.engine.onnx_input_names_no_cache
        )
        engine_input = self.tokens_to_engine_input(input_tokens, onnx_input_names)

        if inputs.session_id is not None:
            # if session_id is provided, we need to set it in engines
            self.engine.session_id = inputs.session_id
            self.multitoken_engine.session_id = inputs.session_id

        postprocessing_kwargs = dict(
            return_logits=inputs.return_logits, streamer=inputs.streamer
        )
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
        self, engine_inputs: List[numpy.ndarray], context: Dict
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Run the forward pass on the engine.

        :param engine_inputs: list of numpy inputs to
            Pipeline engine forward pass
        :return: A tuple of numpy array that contains the
            sequence of generated tokens and a sequence
            of logits for each generated token
        """
        # engine_forward is always called in a threadpool due to batch splitting
        # as such, a new context needs to be created since we are no longer in the
        # main thread. That is why `engine_` is prepended to each of the timer phase
        # names in this context
        with self.timer_manager.new_timer_context(total_inference=False) as timer:
            streamer = context.get("streamer")

            if not self.cache_support_enabled:
                tokens, prompt_logits = self.multitoken_engine(engine_inputs)
                return numpy.array([tokens]), prompt_logits

            else:
                # run the prompt through
                with timer.time(_TextGenerationTimings.PROMPT_PREFILL):
                    tokens, prompt_logits = self.prompt_inference(engine_inputs)

            if streamer is not None:
                streamer.put(numpy.array(tokens))

            # create the generated output
            max_tokens = (
                self.max_generated_tokens
                if self.max_generated_tokens and self.max_generated_tokens > 0
                else 100 * self.sequence_length
            )  # set safety for absolute max generation

            generated_tokens = [tokens[-1]]
            generated_logits = prompt_logits

            with timer.time(_TextGenerationTimings.TOKEN_GENERATION):
                while len(generated_tokens) < max_tokens:
                    with timer.time(_TextGenerationTimings.TOKEN_GENERATION_SINGLE):
                        token, logits = self.autoregressive_inference(tokens)
                    tokens.append(token)
                    generated_tokens.append(token)
                    generated_logits.append(logits)

                    if streamer is not None:
                        streamer.put(numpy.array([token]))

                    if (
                        token == self.tokenizer.eos_token_id
                        and not self.force_max_tokens
                    ):
                        break

            if streamer is not None:
                streamer.end()

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

        if (
            len(tokens) > self.prompt_processing_sequence_length
            and self.enable_multitoken_prefill
        ):
            for engine_inputs in self.engine_inputs_for_prefill(tokens):
                new_token, new_logits = self.multitoken_engine(engine_inputs)
                num_tokens_processed += self.prompt_processing_sequence_length
                prompt_logits.append(new_logits)

        if num_tokens_processed:
            # transfer the cache state from the multi-token engine to the main engine
            self.engine.transfer_cache_state(cache=self.multitoken_engine.kv_cache)

        # prompt size is small, run autoregressive inference to populate kv cache
        run_tokens = [] if num_tokens_processed == 0 else tokens[:num_tokens_processed]

        for token in tokens[num_tokens_processed:]:
            run_tokens.append(token)
            with self.timer_manager.current.time(
                _TextGenerationTimings.PROMPT_PREFILL_SINGLE
            ):
                new_token, new_logits = self.autoregressive_inference(run_tokens)

            prompt_logits.append(new_logits)

        tokens.append(new_token)

        return tokens, prompt_logits

    def autoregressive_inference(
        self,
        tokens: List[int],
    ) -> Tuple[int, numpy.ndarray]:
        """
        An inference run that processes the last token to generate
        a new token and new logits.

        :param tokens: The current context (prompt + generated tokens so far)
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
        positions -= 1
        input_ids = numpy.array([[new_token]])
        causal_mask = create_causal_mask(input_ids, attention_mask)

        # filter out the inputs that are not needed by the engine
        engine_inputs_map = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            positions=positions,
        )
        engine_inputs = [
            engine_inputs_map[name] for name in self.engine.onnx_input_names_no_cache
        ]

        generated_token, generated_logits = self.engine(engine_inputs)

        return generated_token, generated_logits

    def engine_inputs_for_prefill(
        self, tokens: List[int]
    ) -> Generator[List[numpy.ndarray], None, None]:
        """
        Takes a list of tokens and creates a generator
        of engine_inputs for the multitoken engine.

        1. The input tokens first get batched into chunks of
        size self.prompt_processing_sequence_length. This is to
        ensure that they match the expected input size by the
        multitoken engine. Any remaining tokens are discarded.

        2. Every created engine_inputs batch is then created:

            - input_ids: by taking a batch of tokens

            - attention_mask: by creating an appropriate mask,
            that will have the amount of unmasked entries equal to
            the sum of:
                a) the number of tokens in the batch
                (self.prompt_processing_sequence_length)
                b) the number of non-blank cache entries
                (num_non_blank_cache_entries)
            so that the attention_mask properly attends to the
            current input tokens, as well as the previous cache
            entries.

            - positions: derived directly from the input_ids

            - causal_mask: derived from the input_ids and attention_mask

        :param tokens: the list of tokens to process
        :return: a generator of engine inputs
        """

        num_batches = len(tokens) // self.prompt_processing_sequence_length

        token_batches = [
            tokens[
                i
                * self.prompt_processing_sequence_length : (i + 1)
                * self.prompt_processing_sequence_length
            ]
            for i in range(0, num_batches)
        ]

        for idx, token_batch in enumerate(token_batches):
            engine_inputs = []
            num_cached_entries = self.multitoken_engine.num_non_blank_cache_entries
            for name in self.multitoken_engine.onnx_input_names_no_cache:
                if name == "input_ids":
                    engine_input = numpy.array([token_batch])

                elif name == "attention_mask":
                    # create an empty attention mask
                    engine_input = numpy.zeros(
                        (1, self.sequence_length), dtype=numpy.int64
                    )
                    # fill it out with 1s (from the right), so that the number
                    # of unmasked entries is equal to the sum of:
                    engine_input[
                        :,
                        -(
                            # ...the number of current input tokens...
                            self.prompt_processing_sequence_length
                            # ...and the number of the previous cache entries
                            + num_cached_entries
                        ) :,
                    ] = 1
                elif name == "causal_mask":
                    # delay creation of the causal mask
                    continue
                elif name == "positions":
                    if self.prompt_processing_sequence_length == 1:
                        # we need to treat `positions` as if we were in
                        # the autoregressive mode
                        engine_input = numpy.array([[idx]], dtype=numpy.int64)
                    else:
                        engine_input = (
                            numpy.arange(
                                num_cached_entries,
                                num_cached_entries
                                + self.prompt_processing_sequence_length,
                            )
                            .reshape(1, -1)
                            .astype(numpy.int64)
                        )

                engine_inputs.append(engine_input)

            # create the causal mask once we have the input_ids and attention_mask
            if "causal_mask" in self.multitoken_engine.onnx_input_names_no_cache:
                causal_mask = create_causal_mask(
                    input_ids=engine_inputs[0], attention_mask=engine_inputs[1]
                )
                engine_inputs.append(causal_mask)

            yield engine_inputs

    def is_cache_support_enabled(self) -> bool:
        """
        Returns whether the ran model has kv cache or not

        :return: True if the model has kv cache, False otherwise
        """
        return any(default_cached_outputs(self.onnx_file_path))

    def join_engine_outputs(
        self, batch_outputs: List[List[numpy.ndarray]], orig_batch_size: int
    ) -> List[numpy.ndarray]:
        """
        Takes a list of outputs (batches) from the engine
        and joins them into a single output. Asserts that
        the dimensions of the outputs are the same, so that
        they can be concatenated.

        :param batch_outputs: A list of outputs from the engine
        :param orig_batch_size: The original batch size
        :return: A list of joined outputs
        """
        tokens, logits = zip(*batch_outputs)
        if self.cache_support_enabled:
            # if the model has kv cache, we need to account for
            # the fact that the predicted outputs may have
            # different lengths

            # find the longest sequence in the batch of tokens
            max_len = max([token.shape[1] for token in tokens])

            # pad all tokens to the same length
            tokens = [
                pad_to_fixed_length(
                    array=prediction,
                    max_len=max_len,
                    value=self.tokenizer.pad_token_id,
                    axis=1,
                )
                for prediction in tokens
            ]

            # find the longest sequence in the batch of logits
            max_len = max([logits.shape[1] for logits in logits])

            # pad all logits to the same length
            logits = [
                pad_to_fixed_length(array=single_logits, max_len=max_len, axis=1)
                for single_logits in logits
            ]

        tokens = numpy.concatenate(tokens, axis=0)
        logits = numpy.concatenate(logits, axis=0)

        return [tokens, logits]

    @staticmethod
    def causal_mask_input_present(model_path: str) -> bool:
        """
        Check whether the model has causal_mask input present or not.
        In general, the absence of causal_mask input means that the model
        cannot be run through the multitoken engine.

        :param model_path: path to the model
        :return: True if causal_mask input is present, False otherwise
        """
        return any(
            inp.name == "causal_mask"
            for inp in onnx.load(model_path, load_external_data=False).graph.input
        )

    def _reset_engines_cache(self):
        self.engine.reset_kv_cache()
        self.multitoken_engine.reset_kv_cache() if self.multitoken_engine else None
