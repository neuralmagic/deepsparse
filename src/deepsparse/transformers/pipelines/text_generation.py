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

import contextvars
import logging
import os
import warnings
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy
import onnx
from pydantic import BaseModel, Field, validator
from transformers import TextStreamer

from deepsparse import Pipeline
from deepsparse.pipeline import DEEPSPARSE_ENGINE
from deepsparse.transformers.engines import NLDecoderEngine
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.utils.helpers import (
    create_causal_mask,
    generate_session_id,
    pad_to_fixed_length,
    repeat_inputs,
    validate_session_ids,
)
from deepsparse.transformers.utils.timings import TextGenerationTimings
from deepsparse.utils.data import split_engine_inputs
from deepsparse.utils.onnx import default_cached_outputs


_LOGGER = logging.getLogger(__name__)

__all__ = ["TextGenerationPipeline"]

_SESSION_ID = contextvars.ContextVar("_SESSION_ID", default=None)


class TextGenerationInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    sequences: Union[str, List[str]] = Field(
        description="The input sequences to generate the text from.",
    )
    num_generated_predictions: int = Field(
        default=1,
        description="The number of text generations to create from a single prompt. If "
        "the same sequence is given as an input multiple times, the number of generated"
        "the number of generated predictins is equivalent to the number of times the "
        "the sequence is repeated.",
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum number of tokens to generate per output sequence. If no "
        "value is provided, will default to 1024.",
    )
    return_logits: bool = Field(
        default=False,
        description="A flag that indicates whether to return "
        "the logits for the input text sequence and the "
        "generated text sequence. ",
    )
    include_prompt_logits: bool = Field(
        default=False,
        description="A flag that indicates whether to return "
        "the logits for the prompt. If set, prompt_logits are "
        "`prepended` to the logits for the generated text sequence."
        "Note: This flag is only applicable when return_logits "
        "is `True`.",
    )
    session_ids: Union[None, List[str], str] = Field(
        default=None,
        description="A user may set a string identifier(s) "
        "for the kv cache session(s). If None, "
        "and the model is using kv cache, session_id "
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
    callback: Optional[Callable[[Any], Union[bool, Any]]] = Field(
        default=None,
        description="Callable that will be invoked "
        "on each generated token. If the callable returns "
        "`False`, the generation will stop. Default is `None`.",
    )
    stop: Union[None, str, Sequence[str]] = Field(
        default=None,
        description="A string or a list of strings that will be used as"
        " stop tokens. (token generation will stop when any of the stop"
        " tokens is generated). Set to `None` to ignore this parameter."
        " Default is `None`.",
    )

    @validator("session_ids")
    def validate_session_ids(cls, value, values) -> Union[None, List[str]]:
        session_ids = validate_session_ids(session_ids=value, other_attributes=values)
        return session_ids


class TextGenerationOutput(BaseModel):
    sequences: Union[str, List[str], List[List[str]]] = Field(
        description="The generated text sequences.",
    )
    logits: Optional[Any] = Field(  # numpy array, set to Any for FastAPI compatibility
        default=None,
        description="The logits for the generated text sequence."
        "The logits have dimensions "
        "[batch_size, sequence_length, vocab_size]",
    )
    session_ids: Union[None, str, List[str]] = Field(
        default=None, description="A string identifier(s) for the kv cache session."
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

    :param deterministic: if False, the pipeline will sample from
        the probability distribution computed from the logits.
        If True, the pipeline will get the next token by applying
        an argmax function to the logits.
    :param sampling_temperature: the temperature to use when sampling
        from the probability distribution computed from the logits.
        Higher values will result in more random samples. Should
        be greater than 0.0.
    :param sequence_length: sequence length to compile model and tokenizer for.
        This controls the maximum context length of the pipeline. Default is 512
    :param prompt_sequence_length: For large prompts, the prompt is
        processed in chunks of this length. This is to maximize the inference
        speed. By default, this is set to 64.
    :param force_max_tokens: if True, the pipeline will generate the maximum number
        of tokens supplied even if the stop token is reached.
    :param internal_kv_cache: if True, the pipeline will use the deepsparse kv cache
        for caching the model outputs.
    :param kwargs: kwargs to pass to the TransformersPipeline
    """

    def __init__(
        self,
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        prompt_sequence_length: int = 64,
        sequence_length: int = 512,
        force_max_tokens: bool = False,
        internal_kv_cache: bool = True,
        **kwargs,
    ):
        kwargs_engine_type = kwargs.get("engine_type", DEEPSPARSE_ENGINE)

        if internal_kv_cache:
            if kwargs_engine_type != DEEPSPARSE_ENGINE:
                _LOGGER.warning(
                    "`internal_kv_cache` is set to True "
                    "but the chosen `engine_type` "
                    f"is {kwargs_engine_type}. "
                    f"The optimized kv cache management is disabled."
                )
                internal_kv_cache = False

        super().__init__(
            **kwargs,
            sequence_length=sequence_length,
            _delay_engine_initialize=True,
            _delay_overwriting_inputs=True,
        )
        # enable multitoken prefill if
        # - the model graph is supporting it (causal_mask input is present)
        # - prompt_sequence_length != 1 (identical to single-token prefill)
        self.enable_multitoken_prefill = (
            self.causal_mask_input_present(model_path=self.onnx_file_path)
            and prompt_sequence_length > 1
        )

        self.cache_support_enabled = self.is_cache_support_enabled()

        if self.engine_type == DEEPSPARSE_ENGINE:
            if "WAND_OPT_FLAGS" not in os.environ:
                os.environ["WAND_OPT_FLAGS"] = "default,~pyramids"

        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.prompt_sequence_length = prompt_sequence_length
        self.force_max_tokens = force_max_tokens
        self.internal_kv_cache = internal_kv_cache

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
            if (
                self.engine_type == DEEPSPARSE_ENGINE
                and self.sequence_length <= self.prompt_sequence_length
                and self.enable_multitoken_prefill
            ):
                raise ValueError(
                    "Attempting to initialize auxiliary DeepSparse engine to "
                    "process a prompt with a larger processing length. "
                    "However, it is assumed that `prompt_sequence_length` "
                    "is smaller than the `sequence_length`. "
                    "Adjust the `prompt_sequence_length` "
                    "argument accordingly."
                )

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

            # input_ids_length for the multitoken engine is either:
            # - the prompt_sequence_length if the cache support is enabled
            #   (the prompt is processed sequentially at predefined processing length)
            # - the full sequence_length if the cache support is disabled
            #   (the prompt is processed in a single pass, prompts length is fixed at
            #   sequence_length)
            input_ids_length = (
                self.prompt_sequence_length
                if self.cache_support_enabled
                else self.sequence_length
            )

            multitoken_engine = NLDecoderEngine(
                onnx_file_path=self.onnx_file_path,
                engine_type=self.engine_type,
                engine_args=self.engine_args,
                engine_context=self.context,
                sampling_temperature=self.sampling_temperature,
                deterministic=self.deterministic,
                sequence_length=self.sequence_length,
                input_ids_length=input_ids_length,
                tokenizer=self.tokenizer,
                internal_kv_cache=self.internal_kv_cache,
                timer_manager=self.timer_manager,
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
                internal_kv_cache=self.internal_kv_cache,
                timer_manager=self.timer_manager,
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

    def process_inputs(
        self, inputs: TextGenerationInput
    ) -> Tuple[List[numpy.ndarray], Dict[str, Any]]:
        """
        Convert the input schema for the pipeline to the inputs for the engine.

        :param inputs: the input schema for the pipeline
        :return: the inputs for the engine
        """
        if not self.cache_support_enabled and inputs.max_tokens > 1:
            raise ValueError(
                "The model used for inference does not support kv cache. It is "
                "assumed that it maps from the token sequence to predicted logits."
                "Set `max_tokens` to 1 to support that scenario."
            )

        # If the num_generated_predictions > 1, repeat the prompt
        # num_generated_predictions times. Also, update the engine so that deterministic
        # is set to False.
        if inputs.num_generated_predictions > 1:
            if isinstance(inputs.sequences, str):
                inputs.sequences = [inputs.sequences]
            inputs.sequences = repeat_inputs(
                inputs.sequences, inputs.num_generated_predictions
            )
            if self.engine:
                self.engine.deterministic = False
            if self.multitoken_engine:
                self.multitoken_engine.deterministic = False

        if inputs.fixed_sequences_length or not self.cache_support_enabled:
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

        session_ids = inputs.session_ids
        if session_ids is None:
            # session_ids is None, so we need to generate
            # a session id for each input sequence

            # check if _SESSION_ID context is set
            # if it is, use that as the session id

            if _SESSION_ID.get() is not None:
                session_ids = _SESSION_ID.get()
            else:
                num_input_sequences = (
                    len(inputs.sequences) if isinstance(inputs.sequences, list) else 1
                )
                session_ids = [
                    generate_session_id() for _ in range(num_input_sequences)
                ]
        engine_input.append(session_ids)

        context = dict(
            num_generated_predictions=inputs.num_generated_predictions,
            return_logits=inputs.return_logits,
            streamer=inputs.streamer,
            include_prompt_logits=inputs.include_prompt_logits,
            callback=inputs.callback,
            stop=inputs.stop,
            max_tokens=inputs.max_tokens,
        )
        return engine_input, context

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **context
    ) -> TextGenerationOutput:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :return: the output schema for the pipeline
        """
        generated_tokens, generated_logits, session_ids = engine_outputs
        sequences = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        logits = generated_logits if context.get("return_logits") else None
        num_preds = context.get("num_generated_predictions", 1)
        # If the num_generated_predictions > 1, group the generated sequences and return
        # the sequences as a list of lists where each list consists of the generated
        # predictions for a given prompt, and all the lists are in the order matching
        # the order that the prompts were given as inputs.
        if num_preds > 1:
            grouped_seq = [
                sequences[n : n + num_preds]
                for n in range(0, len(sequences), num_preds)
            ]
            sequences = grouped_seq

        logits = generated_logits if context.get("return_logits") else None

        return TextGenerationOutput(
            sequences=sequences, logits=logits, session_ids=session_ids.tolist()
        )

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], context: Dict
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Run the forward pass on the engine.

        :param engine_inputs: List of numpy inputs to
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

            # engine_inputs is a list of numpy arrays plus additional
            # session_id string. We need to pop the session_id string
            # and from the engine_inputs. The session_id will be used
            # seperately to keep track of the appropriate kv cache session
            # (if kv cache is enabled)
            session_id = engine_inputs.pop(
                next(
                    idx
                    for idx, item in enumerate(engine_inputs)
                    if isinstance(item, str)
                )
            )

            assert isinstance(
                session_id, str
            ), "Session id must be a string not {}".format(type(session_id))

            if not self.cache_support_enabled:
                session_id = None
                tokens, prompt_logits = self.multitoken_engine(
                    engine_inputs, session_id
                )
                return (
                    numpy.array([tokens]),
                    prompt_logits,
                    numpy.array([session_id]),
                )

            else:
                # run the prompt through
                with timer.time(TextGenerationTimings.PROMPT_PREFILL):
                    tokens, prompt_logits = self.prompt_inference(
                        engine_inputs, session_id
                    )

            if streamer is not None:
                streamer.put(numpy.array(tokens))

            # create the generated output
            max_tokens = context.get("max_tokens", 0)
            max_tokens = max_tokens if max_tokens > 0 else (100 * self.sequence_length)

            # last prompt token is the first generated token
            # add it to generated tokens, and the logits
            generated_tokens = [tokens[-1]]
            generated_logits = (
                prompt_logits
                if context.get("include_prompt_logits")
                else [prompt_logits[-1]]
            )
            callback = context.get("callback")
            stop = context.get("stop")

            with timer.time(TextGenerationTimings.TOKEN_GENERATION):
                while len(generated_tokens) < max_tokens:
                    with timer.time(TextGenerationTimings.TOKEN_GENERATION_SINGLE):
                        token, logits = self.autoregressive_inference(
                            tokens, session_id
                        )
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

                    if self._stop_token_generated(token, stop_tokens=stop):
                        _LOGGER.debug(
                            "Stop token %s generated. Stopping generation."
                            % self.tokenizer.decode(token)
                        )
                        break

                    if callback is not None and callback(token) is False:
                        _LOGGER.debug(
                            "callback %s returned False, stopping generation."
                            % callback.__qualname__
                        )
                        break
            # Run the autoregressive inference only to put the
            # kv cache entry for the last generated token into the
            # kv cache
            self.autoregressive_inference(tokens, session_id)
            if streamer is not None:
                streamer.end()

        return (
            numpy.array([generated_tokens]),
            numpy.concatenate(generated_logits, axis=1),
            numpy.array([session_id]),
        )

    def prompt_inference(
        self, engine_inputs: List[numpy.ndarray], session_id: str
    ) -> Tuple[List[int], List[numpy.ndarray]]:
        """
        An inference run that processes the prompt through the
        model to generate the new token and logits

        :param engine_inputs: The prompt (context) represented by a
            list of numpy inputs to the engine
        :param session_id: The session id to run the inference under
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

        if len(tokens) > self.prompt_sequence_length and self.enable_multitoken_prefill:

            self.synchronize_engines(session_id)
            tokens = (
                self._remove_bos_token_if_applicable(tokens)
                if self.multitoken_engine.kv_cache_storage.has_session(session_id)
                else tokens
            )

            for engine_inputs in self.engine_inputs_for_prefill(tokens, session_id):
                new_token, new_logits = self.multitoken_engine(
                    engine_inputs, session_id
                )
                num_tokens_processed += self.prompt_sequence_length
                prompt_logits.append(new_logits)

        # prompt size is small, run autoregressive inference to populate kv cache
        run_tokens = [] if num_tokens_processed == 0 else tokens[:num_tokens_processed]

        self.synchronize_engines(session_id)
        tokens = (
            self._remove_bos_token_if_applicable(tokens)
            if self.engine.kv_cache_storage.has_session(session_id)
            and not num_tokens_processed
            else tokens
        )

        for token in tokens[num_tokens_processed:]:
            run_tokens.append(token)
            with self.timer_manager.current.time(
                TextGenerationTimings.PROMPT_PREFILL_SINGLE
            ):
                new_token, new_logits = self.autoregressive_inference(
                    run_tokens, session_id
                )

            prompt_logits.append(new_logits)

        tokens.append(new_token)

        return tokens, prompt_logits

    def autoregressive_inference(
        self,
        tokens: List[int],
        session_id: str,
    ) -> Tuple[int, numpy.ndarray]:
        """
        An inference run that processes the last token to generate
        a new token and new logits.

        :param tokens: The current context (prompt + generated tokens so far)
        :param session_id: The session id to run the inference under
        :return: The new, generated token and the logits for the new token
            (with dimensions ['batch_size', 'num_tokens', 'vocab_size'])
        """
        num_total_processed_tokens = self.engine.total_num_processed_tokens(session_id)

        new_token = tokens[-1]
        # padding is added to left, so attention mask is 1s from the
        # right up to the number of total tokens (prompt + generated)
        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)

        num_attention_entries_to_unmask = min(
            num_total_processed_tokens + 1, self.sequence_length
        )  # cap by seq len
        attention_mask[:, -num_attention_entries_to_unmask:] = 1

        positions = numpy.array([[num_total_processed_tokens]], dtype=numpy.int64)
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

        generated_token, generated_logits = self.engine(engine_inputs, session_id)
        _LOGGER.debug(f"position {positions}, token: {input_ids}")
        return generated_token, generated_logits

    def engine_inputs_for_prefill(
        self, tokens: List[int], session_id: str
    ) -> Generator[List[numpy.ndarray], None, None]:
        """
        Takes a list of tokens and creates a generator
        of engine_inputs for the multitoken engine.

        1. The input tokens first get batched into chunks of
        size self.prompt_sequence_length. This is to
        ensure that they match the expected input size by the
        multitoken engine. Any remaining tokens are discarded.

        2. Every created engine_inputs batch is then created:

            - input_ids: by taking a batch of tokens

            - attention_mask: by creating an appropriate mask,
            that will have the amount of unmasked entries equal to
            the sum of:
                a) the number of tokens in the batch
                (self.prompt_sequence_length)
                b) the number of processed tokens so far
                (num_total_processed_tokens)
            so that the attention_mask properly attends to the
            current input tokens, as well as the previous cache
            entries.

            - positions: derived directly from the input_ids

            - causal_mask: derived from the input_ids and attention_mask

        :param tokens: the list of tokens to process
        :param session_id: the session id to run the inference under
        :return: a generator of engine inputs
        """

        num_batches = len(tokens) // self.prompt_sequence_length

        token_batches = [
            tokens[
                i * self.prompt_sequence_length : (i + 1) * self.prompt_sequence_length
            ]
            for i in range(0, num_batches)
        ]

        for idx, token_batch in enumerate(token_batches):
            engine_inputs = []
            num_total_processed_tokens = (
                self.multitoken_engine.total_num_processed_tokens(session_id)
            )

            for name in self.multitoken_engine.onnx_input_names_no_cache:
                if name == "input_ids":
                    engine_input = numpy.array([token_batch])

                elif name == "attention_mask":
                    # create an empty attention mask
                    engine_input = numpy.zeros(
                        (1, self.sequence_length), dtype=numpy.int64
                    )
                    num_attention_entries_to_unmask = min(
                        num_total_processed_tokens + self.prompt_sequence_length,
                        self.sequence_length,
                    )
                    engine_input[:, -num_attention_entries_to_unmask:] = 1

                elif name == "causal_mask":
                    # delay creation of the causal mask
                    continue
                elif name == "positions":
                    if self.prompt_sequence_length == 1:
                        # we need to treat `positions` as if we were in
                        # the autoregressive mode
                        engine_input = numpy.array(
                            [[num_total_processed_tokens]], dtype=numpy.int64
                        )
                    else:
                        engine_input = (
                            numpy.arange(
                                num_total_processed_tokens,
                                num_total_processed_tokens
                                + self.prompt_sequence_length,
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

    def synchronize_engines(self, session_id: str):
        """
        Make sure that the existing engines are in sync i.e.
        they contain the newest version of kv cache session with
        the given session id.

        :param session_id: the session id of the session to synchronize
        """
        engine_session = self.engine.kv_cache_storage.get(session_id)
        if self.multitoken_engine:
            multitoken_session = self.multitoken_engine.kv_cache_storage.get(session_id)
        else:
            return

        if engine_session is None and multitoken_session:
            self.engine.transfer_cache_session(multitoken_session)
        elif engine_session and multitoken_session is None:
            self.multitoken_engine.transfer_cache_session(engine_session)
        else:
            pass

    def is_cache_support_enabled(self) -> bool:
        """
        Returns whether the ran model has kv cache or not

        :return: True if the model has kv cache, False otherwise
        """
        return any(default_cached_outputs(self.onnx_file_path))

    def split_engine_inputs(
        self, items: List[Union[numpy.ndarray, List[str]]], batch_size: int
    ) -> Tuple[List[List[numpy.ndarray]], int]:
        """
        Custom implementation of splitting the engine inputs that takes into
        account the fact that the `items` contain additionally a list of
        session_ids, that need to be distributed across the batches.

        :param items: list of numpy arrays to split (plus list of session_ids)
        :param batch_size: size of each batch to split into
        :return: list of batches, where each batch is a list of numpy arrays
            (plus session_ids), as well as the total batch size
        """
        # extract the session_ids from the items
        session_ids = next((item for item in items if isinstance(item, list)), None)
        items = [item for item in items if not isinstance(item, list)]

        batches, orig_batch_size = split_engine_inputs(items, batch_size)

        # distribute session_ids across batches
        batches_w_session_ids = [
            batch + [session_ids[i]] for i, batch in enumerate(batches)
        ]

        return batches_w_session_ids, orig_batch_size

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
        tokens, logits, session_ids = zip(*batch_outputs)
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
        session_ids = numpy.concatenate(session_ids, axis=0)

        return [tokens, logits, session_ids]

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

    @contextmanager
    def session(self, session_id: Optional[str] = None) -> Callable[[Any, Any], Any]:
        """
        A context manager that returns existing pipeline instance
        with the session_id already set.
        """

        if session_id is None:
            session_id = generate_session_id()

        # set session_id contextvar
        token = _SESSION_ID.set([session_id])
        yield
        # reset session_id contextvar
        _SESSION_ID.reset(token)

    def _stop_token_generated(
        self, token, stop_tokens: Union[None, str, Sequence[str]]
    ) -> bool:
        if stop_tokens is None:
            return False

        decoded_token = self.tokenizer.decode(token)
        decoded_token = (
            decoded_token if decoded_token.isspace() else decoded_token.strip()
        )
        return decoded_token in stop_tokens

    def _remove_bos_token_if_applicable(self, tokens: List[int]) -> List[int]:
        if hasattr(self.tokenizer, "add_bos_token"):
            return tokens[1:]
        return tokens
