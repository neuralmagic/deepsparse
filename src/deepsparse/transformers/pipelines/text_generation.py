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

import copy
import datetime
import logging
import os
import pathlib
import warnings
from enum import Enum
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
from pydantic import BaseModel, Field
from transformers import GenerationConfig

from deepsparse import Pipeline
from deepsparse.pipeline import DEEPSPARSE_ENGINE
from deepsparse.transformers.engines import NLDecoderEngine
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.utils import DecoderKVCache
from deepsparse.transformers.utils.helpers import (
    check_and_return_generation_config,
    create_causal_mask,
    initialize_kv_cache_state,
    override_config,
    pad_to_fixed_length,
    prepends_bos_token,
    process_generation_config,
    repeat_inputs,
    set_generated_length,
)
from deepsparse.transformers.utils.timings import TextGenerationTimings
from deepsparse.transformers.utils.token_generator import TokenGenerator
from deepsparse.utils.onnx import default_cached_outputs


_LOGGER = logging.getLogger(__name__)

__all__ = ["TextGenerationPipeline"]


# Based off of https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig # noqa E501
class GenerationDefaults:
    # Parameters that control the length of the output
    max_length = None
    max_new_tokens = 100
    # Parameters that control the generation strategy used
    do_sample = False
    # Parameters for manipulation of the model output logits
    temperature = 1.0
    top_k = 50
    top_p = 1.0
    repetition_penalty = 1.0
    # Parameters that define the outputs
    num_return_sequences = 1
    output_scores = False


class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    TIME = "time"
    CALLBACK = "callback"
    CAPACITY = "capacity"
    MAX_NEW_TOKENS = "max_new_tokens"


class TextGenerationInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    sequences: Union[str, List[str]] = Field(
        alias="prompt",
        description="The input sequences to generate the text from.",
    )
    return_input_tokens: bool = Field(
        default=False,
        description="A flag that indicates whether to return " "the input_tokens. ",
    )
    include_prompt_logits: bool = Field(
        default=False,
        description="A flag that indicates whether to return "
        "the logits for the prompt. If set, prompt_logits are "
        "`prepended` to the logits for the generated text sequence."
        "Note: This flag is only applicable when output_scores "
        "is `True`.",
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
    streaming: bool = Field(
        default=False,
        description="Whether to stream the results back as they are generated. If "
        "True, then the results are returned as a generator object which yields "
        "the results as they are generated. If False, then the results are returned "
        "as a list after it has completed.",
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

    presence_penalty: Optional[float] = Field(
        default=0.0,
        description="Penalty applied for generating new token. Any existing"
        " token results in the subtraction of its corresponding logit value."
        " Default set to 0.0",
    )

    generation_config: Union[None, str, pathlib.Path, Dict, GenerationConfig] = Field(
        default=None,
        description="GenerationConfig file consisting of parameters used to control "
        "sequences generated for each prompt. The current supported parameters are: "
        "max_length, max_new_tokens, num_return_sequences, output_scores, top_p, "
        "top_k, repetition_penalty, do_sample, temperature. If None is provided, "
        "deepsparse defaults will be used. For all other input types, HuggingFace "
        "defaults for GenerationConfig will be used. ",
    )

    generation_kwargs: Optional[Dict] = Field(
        default=None,
        description="Any arguments to override generation_config arguments. Refer to "
        "the generation_config argument for a full list of supported variables.",
    )


class GeneratedText(BaseModel):
    text: str = Field(
        description="The generated sequence for a given prompt. If "
        "streaming is enabled, this will be the next generated token."
    )
    score: Optional[Any] = Field(
        default=None,
        description="The score for the generated token or sequence. "
        "The scores have the shape [sequence_length, vocab_size]",
    )
    finished: bool = Field(description="Whether generation has stopped.")
    finished_reason: Optional[str] = Field(
        default=None,
        description="The reason for generation to stop. "
        "Defined by FinishReason. One of stop, length, or time.",
    )


# TODO: Pydantic aliases allow assignment but not reference. Still need to update.
class TextGenerationOutput(BaseModel):
    created: datetime.datetime = Field(description="Time of inference creation.")
    prompts: Union[str, List[str]] = Field(
        description="Prompts used for the sequence generation. For multiple input "
        "prompts, a list of prompts is returned"
    )
    generations: Union[List[GeneratedText], List[List[GeneratedText]]] = Field(
        description="For a single prompt, a single list of GeneratedText is returned. "
        "If multiple prompts are given, a list of GeneratedText is returned for each "
        "prompt provided. If streamng is enabled, the next generated token is returned."
        "Otherwise, the full generated sequence is returned."
    )
    input_tokens: Optional[
        Any
    ] = Field(  # dictionary mapping "token_ids" and "attention_mask" to numpy arrays
        default=None,
        description="The output of the tokenizer."
        "Dictionary containing token_ids and attention_mask, "
        "both mapping to arrays of size "
        "[batch_size, sequence_length]",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


@Pipeline.register(
    task="text_generation",
    task_aliases=["opt", "bloom"],
)
class TextGenerationPipeline(TransformersPipeline):
    """
    Pipeline for text generation tasks.

    :param sequence_length: sequence length to compile model and tokenizer for.
        This controls the maximum context length of the pipeline. Default is 1024
    :param prompt_sequence_length: For large prompts, the prompt is
        processed in chunks of this length. This is to maximize the inference
        speed. By default, this is set to 64.
    :param force_max_tokens: if True, the pipeline will generate the maximum number
        of tokens supplied even if the stop token is reached.
    :param internal_kv_cache: if True, the pipeline will use the deepsparse kv cache
        for caching the model outputs.
    :param generation_config: config file consisting of parameters used to control
        sequences generated for each prompt. The current supported parameters are:
        max_length, max_new_tokens, num_return_sequences, output_scores, top_p,
        top_k, repetition_penalty, do_sample, temperature. If None is provided,
        deepsparse defaults will be used. For all other input types, HuggingFace
        defaults for GenerationConfig will be used.
    :param kwargs: kwargs to pass to the TransformersPipeline
    """

    def __init__(
        self,
        sequence_length: int = 1024,
        prompt_sequence_length: int = 16,
        force_max_tokens: bool = False,
        internal_kv_cache: bool = True,
        generation_config: Union[str, pathlib.Path, Dict, GenerationConfig] = None,
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

        # the current requirement on the deepsparse engine
        # is that prompt_sequence_length
        # must be 1 or a multiple of four.
        # for simplicity let's extend this requirement to all engines
        if (prompt_sequence_length % 4 != 0) and (prompt_sequence_length != 1):
            raise ValueError(
                f"prompt_sequence_length must be 1 or multiple of 4. "
                f"prompt_sequence_length is {prompt_sequence_length}"
            )
        self.prompt_sequence_length = prompt_sequence_length
        self.force_max_tokens = force_max_tokens
        self.internal_kv_cache = internal_kv_cache

        # override tokenizer to pad to left
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.engine, self.multitoken_engine = self.initialize_engines()

        # auxiliary flag for devs to enable debug mode for the pipeline
        self._debug = False
        self.generation_config = process_generation_config(generation_config)
        if self.generation_config:
            _LOGGER.info(
                "Generation config provided for pipline. This will be used "
                "for all inputs unless an input-specific config is provided. "
            )

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
                    "Creation of an auxiliary engine for "
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
                sequence_length=self.sequence_length,
                input_ids_length=input_ids_length,
                internal_kv_cache=self.internal_kv_cache,
                timer_manager=self.timer_manager,
            )

        if self.cache_support_enabled:
            engine = NLDecoderEngine(
                onnx_file_path=self.onnx_file_path,
                engine_type=self.engine_type,
                engine_args=self.engine_args,
                engine_context=self.context,
                sequence_length=self.sequence_length,
                input_ids_length=1,
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

    def parse_inputs(self, *args, **kwargs) -> TextGenerationInput:
        """

        :param args: in line argument can only have 1, must either be
            a complete TextGenerationInput object or `sequences` for
            a TextGenerationInput
        :param kwargs: if a TextGenerationInput is not provided, then
            these kwargs will be used to instantiate one
        :return: parsed TextGenerationInput object
        """
        if "sequences" in kwargs and "prompt" not in kwargs:
            # support prompt and sequences interchangeably
            kwargs["prompt"] = kwargs["sequences"]

        if (
            args
            and not isinstance(args[0], TextGenerationInput)
            and "prompt" not in kwargs
            and "sequences" not in kwargs
        ):
            # assume first argument is "sequences" (prompt) by default
            kwargs["prompt"] = args[0]
            args = args[1:]

        if kwargs:
            generation_kwargs = kwargs.get("generation_kwargs", {})
            for k, v in kwargs.items():
                if not generation_kwargs.get(k) and hasattr(GenerationDefaults, k):
                    generation_kwargs[k] = v

            kwargs["generation_kwargs"] = generation_kwargs

        return super().parse_inputs(*args, **kwargs)

    def process_inputs(
        self, inputs: TextGenerationInput
    ) -> Tuple[List[numpy.ndarray], Dict[str, Any]]:
        """
        Convert the input schema for the pipeline to the inputs for the engine.

        :param inputs: the input schema for the pipeline
        :return: the inputs for the engine
        """
        generation_config = check_and_return_generation_config(
            self.generation_config, inputs.generation_config, GenerationDefaults()
        )

        generation_config = override_config(inputs.generation_kwargs, generation_config)

        self.streaming = inputs.streaming
        if not self.cache_support_enabled and generation_config.max_length > 1:
            raise ValueError(
                "The model used for inference does not support kv cache. It is "
                "assumed that it maps from the token sequence to predicted logits."
                "Set `max_length` to 1 to support that scenario."
            )

        # If the num_return_sequences > 1, repeat the prompt
        # num_return_sequences times.
        original_inputs = inputs.sequences
        if generation_config.num_return_sequences > 1:
            if isinstance(inputs.sequences, str):
                inputs.sequences = [inputs.sequences]
            inputs.sequences = repeat_inputs(
                inputs.sequences, generation_config.num_return_sequences
            )

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

        context = dict(
            prompts=original_inputs,
            streaming=inputs.streaming,
            return_input_tokens=inputs.return_input_tokens,
            input_tokens=input_tokens,
            generation_config=generation_config,
            include_prompt_logits=inputs.include_prompt_logits,
            callback=inputs.callback,
            stop=inputs.stop,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            presence_penalty=inputs.presence_penalty,
            frequency_penalty=generation_config.repetition_penalty,
        )

        return engine_input, context

    def _create_generated_text_output(
        self,
        sequence: str,
        finish_reason: Optional[FinishReason] = None,
        logits: Optional[numpy.array] = None,
    ):
        if finish_reason:
            return GeneratedText(
                text=sequence,
                score=logits,
                finished=True,
                finished_reason=finish_reason.value,
            )
        return GeneratedText(
            text=sequence,
            score=logits,
            finished=False,
        )

    def _stream_engine_outputs(
        self, engine_outputs, prompts, generation_config, **kwargs
    ):
        for output in engine_outputs:
            (
                generated_tokens,
                generated_logits,
                finished_reason,
                past_tokens_queue,
            ) = output
            logits = generated_logits if generation_config.output_scores else None
            from transformers import LlamaTokenizer, LlamaTokenizerFast

            if isinstance(self.tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
                # temporary fix for LLama2/Mistral/... models
                generated_string = self._generate_streamed_text_from_past_tokens(
                    generated_tokens, past_tokens_queue
                )
            else:
                generated_string = self.tokenizer.batch_decode(generated_tokens)[0]
            generation = self._create_generated_text_output(
                generated_string,
                finished_reason[0],
                logits,
            )
            # Add session_id to schema if it exists
            #  more relevant for `ChatPipeline`
            schema_kwargs = (
                {"session_ids": session_ids}
                if (session_ids := kwargs.get("session_ids"))
                else {}
            )
            yield self.output_schema(
                created=datetime.datetime.now(),
                prompts=prompts,
                generations=[generation],
                **schema_kwargs,
            )

    def _generate_streamed_text_from_past_tokens(
        self, generated_tokens: numpy.ndarray, past_tokens_queue: List[int]
    ) -> str:
        """
        An auxiliary method that helps to properly generate the streamed text.
        Some models like llama2 and mistral are using LlamaTokenizer which is
        based on SentencePiece tokenizer. This specific tokenizer doesn't seem
        to output appropriate prefix spaces when decoding token by token.
        One can make it work if the previously generated tokens are included.
        This allows the tokenizer to figure out that the appropriate spaces
        from last n consecutive tokens.

        :param generated_tokens: the generated tokens from the engine
        :param past_tokens_queue: the queue of last n tokens (n is the
            original prompt length in tokens)
        :return: the generated string
        """
        string_from_n_tokens = self.tokenizer.decode(
            past_tokens_queue, skip_special_tokens=True
        )
        past_tokens_queue.append(generated_tokens[0])
        string_from_n_plus_1_tokens = self.tokenizer.decode(
            past_tokens_queue, skip_special_tokens=True
        )
        past_tokens_queue.pop(0)
        return string_from_n_plus_1_tokens[len(string_from_n_tokens) :]

    def process_engine_outputs(
        self, engine_outputs: List[Union[numpy.ndarray, FinishReason]], **kwargs
    ) -> TextGenerationOutput:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :return: the output schema for the pipeline
        """

        generation_config = kwargs.get("generation_config")
        prompts = kwargs.get("prompts")
        streaming = kwargs.get("streaming")

        if streaming:
            return self._stream_engine_outputs(
                engine_outputs, prompts, generation_config
            )

        if self._debug:
            (
                generated_tokens,
                generated_logits,
                finished_reason,
                kv_cache_state,
                total_num_processed_tokens,
            ) = list(*engine_outputs)
        else:
            generated_tokens, generated_logits, finished_reason = list(*engine_outputs)
        sequences = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        logits = generated_logits if generation_config.output_scores else None

        num_preds = generation_config.num_return_sequences
        finished_reason = [f[0] for f in finished_reason]

        if logits is not None:
            generations = list(
                self.executor.map(
                    self._create_generated_text_output,
                    sequences,
                    finished_reason,
                    logits,
                )
            )
        else:
            generations = list(
                self.executor.map(
                    self._create_generated_text_output, sequences, finished_reason
                )
            )

        # If the num_return_sequences > 1, group the generations and return
        # them as a list of lists where each list consists of the generated
        # predictions for a given prompt, and all the lists are in the order matching
        # the order that the prompts were given as inputs.
        if num_preds > 1:
            grouped_generations = [
                generations[n : n + num_preds]
                for n in range(0, len(generations), num_preds)
            ]
            generations = grouped_generations

        input_tokens = (
            kwargs.get("input_tokens") if kwargs.get("return_input_tokens") else None
        )

        outputs = dict(
            created=datetime.datetime.now(),
            prompts=prompts,
            generations=generations,
            input_tokens=input_tokens,
        )

        if "session_ids" in kwargs:
            outputs["session_ids"] = kwargs["session_ids"]

        if self._debug:
            debug_params = dict(
                kv_cache_state=kv_cache_state,
                total_num_processed_tokens=total_num_processed_tokens,
            )
            outputs.update(debug_params)

        return self.output_schema(**outputs)

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], context: Dict
    ) -> Union[
        Tuple[numpy.ndarray, numpy.ndarray, List[FinishReason]],
        Tuple[numpy.ndarray, numpy.ndarray, List[FinishReason], DecoderKVCache],
    ]:
        """
        Run the forward pass on the engine.

        :param engine_inputs: list of numpy inputs to
            Pipeline engine forward pass
        :return: A tuple of:
            - numpy array that contains the sequence
                of generated tokens
            - numpy array that cointains the sequence of
                logits for each generated token
        """
        # engine_forward is always called in a threadpool due to batch splitting
        # as such, a new context needs to be created since we are no longer in the
        # main thread. That is why `engine_` is prepended to each of the timer phase
        # names in this context

        with self.timer_manager.new_timer_context(total_inference=False) as timer:
            finished_reason = []
            streaming = context.get("streaming")
            generation_config = context.get("generation_config")
            deterministic = not generation_config.do_sample

            if not self.cache_support_enabled:
                prompt_logits = self.multitoken_engine(engine_inputs)
                token_generator = TokenGenerator(
                    logits_shape=prompt_logits[-1].shape[-1],
                    deterministic=deterministic,
                    sampling_temperature=generation_config.temperature,
                    **context,
                )
                for prompt_logit in prompt_logits:
                    token_generator.generate(prompt_logit)
                yield numpy.array([token_generator.tokens]), prompt_logits, [
                    FinishReason.LENGTH
                ]
                return

            else:
                # run the prompt through
                with timer.time(TextGenerationTimings.PROMPT_PREFILL):
                    prompt_logits, session = self.prompt_inference(engine_inputs)

            tokens = engine_inputs[0][engine_inputs[1].nonzero()].tolist()
            # copy the tokens so that we can use them for streaming
            past_tokens_queue = copy.copy(tokens)

            token_generator = TokenGenerator(
                logits_shape=prompt_logits[-1].shape[-1],
                tokens=tokens,
                deterministic=deterministic,
                sampling_temperature=generation_config.temperature,
                **context,
            )
            token_generator.generate(prompt_logits[-1][0, -1, :])

            # last prompt token is the first generated token
            # add it to generated tokens, and the logits
            generated_tokens = [token_generator.tokens[-1]]

            generated_logits = (
                prompt_logits
                if context.get("include_prompt_logits")
                else [prompt_logits[-1]]
            )

            callback = context.get("callback")
            stop = context.get("stop")

            max_tokens, length_finish_reason = set_generated_length(
                max_length=generation_config.max_length,
                prompt_tokens_length=len(generated_tokens),
                max_new_tokens=generation_config.max_new_tokens,
                sequence_length=self.sequence_length,
                prompt_sequence_length=self.prompt_sequence_length,
                finish_reason_choices=FinishReason,
            )

            with timer.time(TextGenerationTimings.TOKEN_GENERATION):
                if len(generated_tokens) < max_tokens:
                    if streaming:
                        yield (
                            numpy.array([generated_tokens[-1]]),
                            numpy.array([generated_logits[-1]]),
                            [None],
                            past_tokens_queue,
                        )

                while len(generated_tokens) < max_tokens:
                    with timer.time(TextGenerationTimings.TOKEN_GENERATION_SINGLE):
                        logits = self.autoregressive_inference(
                            tokens=token_generator.tokens, kv_cache=session
                        )
                        token = token_generator.generate(logits=logits[0, -1, :])

                    generated_tokens.append(token)
                    generated_logits.append(logits)

                    if session.total_num_processed_tokens >= session.capacity:
                        # if the kv cache is full, stop generation
                        finished_reason.append(FinishReason.CAPACITY)
                        break

                    if (
                        token == self.tokenizer.eos_token_id
                        and not self.force_max_tokens
                    ):
                        finished_reason.append(FinishReason.STOP)
                        break

                    if self._stop_token_generated(token, stop_tokens=stop):
                        _LOGGER.debug(
                            "Stop token %s generated. Stopping generation."
                            % self.tokenizer.decode(token)
                        )
                        finished_reason.append(FinishReason.STOP)
                        break

                    if callback is not None and callback(token) is False:
                        _LOGGER.debug(
                            "callback %s returned False, stopping generation."
                            % callback.__qualname__
                        )
                        finished_reason.append(FinishReason.CALLBACK)
                        break

                    if len(generated_tokens) == max_tokens:
                        finished_reason.append(length_finish_reason)
                        break

                    if streaming:
                        yield (
                            numpy.array([token]),
                            numpy.array([logits]),
                            [None],
                            past_tokens_queue,
                        )

                # Run the autoregressive inference only to put the
                # kv cache entry for the last generated token into the
                # kv cache
                self.autoregressive_inference(
                    tokens=token_generator.tokens, kv_cache=session
                )
                if streaming:
                    # when no new tokens are generated
                    if len(finished_reason) == 0:
                        yield (
                            numpy.array([generated_tokens]),
                            numpy.concatenate(generated_logits, axis=1),
                            [FinishReason.LENGTH],
                            past_tokens_queue,
                        )
                    else:
                        yield (
                            numpy.array([token]),
                            numpy.array([logits]),
                            [finished_reason[-1]],
                            past_tokens_queue,
                        )

        if not streaming:
            # when no new tokens are generated
            if len(finished_reason) == 0:
                finished_reason.append(FinishReason.LENGTH)

            if self._debug:
                returns = (
                    numpy.array([generated_tokens]),
                    numpy.concatenate(generated_logits, axis=1),
                    finished_reason,
                    [session],
                )
            else:
                returns = (
                    numpy.array([generated_tokens]),
                    numpy.concatenate(generated_logits, axis=1),
                    finished_reason,
                )

            yield returns

    def prompt_inference(
        self,
        engine_inputs: List[numpy.ndarray],
    ) -> Tuple[List[numpy.ndarray], DecoderKVCache]:
        """
        An inference run that processes the prompt through the
        model to generate the new token and logits

        :param engine_inputs: the prompt (context) represented by a
            list of numpy inputs to the engine
        :return: A tuple of:
            - The logits generated from the prompt (with dimensions
            ['batch_size', 'num_tokens', 'vocab_size'])
            - The kv cache session for this inference run
        """
        # get tokens by attention mask
        tokens = engine_inputs[0][engine_inputs[1].nonzero()].tolist()

        prompt_logits = []
        num_tokens_processed = 0

        session = self.get_kv_cache_decoder(engine_inputs)

        if len(tokens) > self.prompt_sequence_length and self.enable_multitoken_prefill:
            for engine_inputs in self.engine_inputs_for_prefill(
                tokens, kv_cache=session
            ):
                new_logits = self.multitoken_engine(engine_inputs, kv_cache=session)
                num_tokens_processed += self.prompt_sequence_length
                prompt_logits.append(new_logits)

        session.set_capacity(self.sequence_length - 1)

        # prompt size is small, run autoregressive inference to populate kv cache
        run_tokens = [] if num_tokens_processed == 0 else tokens[:num_tokens_processed]

        for token in tokens[num_tokens_processed:]:
            run_tokens.append(token)
            with self.timer_manager.current.time(
                TextGenerationTimings.PROMPT_PREFILL_SINGLE
            ):
                new_logits = self.autoregressive_inference(run_tokens, session)

            prompt_logits.append(new_logits)

        return prompt_logits, session

    def autoregressive_inference(
        self,
        tokens: List[int],
        kv_cache: DecoderKVCache,
    ) -> Tuple[int, numpy.ndarray]:
        """
        An inference run that processes the last token to generate
        a new token and new logits.

        :param tokens: The current context (prompt + generated tokens so far)
        :return: The new, generated token and the logits for the new token
            (with dimensions ['batch_size', 'num_tokens', 'vocab_size'])
        """

        num_total_processed_tokens = kv_cache.total_num_processed_tokens
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
        generated_logits = self.engine(engine_inputs, kv_cache)

        return generated_logits

    def engine_inputs_for_prefill(
        self, tokens: List[int], kv_cache: DecoderKVCache
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
            Note: the aformentioned sum must be capped
            by the sequence length, as the maximum shape of the
            attention mask is [batch_size, sequence_length].

            - positions: derived directly from the input_ids

            - causal_mask: derived from the input_ids and attention_mask

        :param tokens: the list of tokens to process
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
            num_total_processed_tokens = kv_cache.total_num_processed_tokens
            engine_inputs = []
            for name in self.multitoken_engine.onnx_input_names_no_cache:
                if name == "input_ids":
                    engine_input = numpy.array([token_batch])

                elif name == "attention_mask":
                    # create an empty attention mask
                    engine_input = numpy.zeros(
                        (1, self.sequence_length), dtype=numpy.int64
                    )
                    # calculate the number of entries in attention mask
                    # that should be set to 1
                    num_attention_entries_to_unmask = min(
                        num_total_processed_tokens + self.prompt_sequence_length,
                        self.sequence_length,
                    )
                    engine_input[:, -num_attention_entries_to_unmask:] = 1
                elif name == "causal_mask":
                    # delay creation of the causal mask
                    continue
                elif name == "positions":
                    engine_input = (
                        numpy.arange(
                            num_total_processed_tokens,
                            num_total_processed_tokens + self.prompt_sequence_length,
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
        self,
        batch_outputs: List[List[Union[numpy.ndarray, FinishReason]]],
        orig_batch_size: int,
        **kwargs,
    ) -> List[Union[numpy.ndarray, FinishReason]]:
        """
        Takes a list of outputs (batches) from the engine
        and joins them into a single output. Asserts that
        the dimensions of the outputs are the same, so that
        they can be concatenated.

        :param batch_outputs: A list of outputs from the engine
        :param orig_batch_size: The original batch size
        :return: A list of joined outputs
        """
        streaming = kwargs.get("streaming")
        if streaming:
            for batch in batch_outputs:
                for outputs in batch:
                    yield outputs
        else:
            batch_outputs = [list(*b) for b in batch_outputs]
            if self._debug:
                tokens, logits, finish_reason, debug = zip(*batch_outputs)
            else:
                tokens, logits, finish_reason = zip(*batch_outputs)
                debug = None

            if self.cache_support_enabled:
                # if the model has kv cache, we need to account for
                # the fact that the predicted outputs may have
                # different lengths

                # find the longest sequence in the batch of tokens
                max_len = max(token.shape[1] for token in tokens)

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
                max_len = max(logits.shape[1] for logits in logits)

                # pad all logits to the same length
                logits = [
                    pad_to_fixed_length(array=single_logits, max_len=max_len, axis=1)
                    for single_logits in logits
                ]

            tokens = numpy.concatenate(tokens, axis=0)
            logits = numpy.concatenate(logits, axis=0)

            if debug:
                sessions = debug[0]
                kv_cache_state = numpy.stack(
                    [session.cached_inputs for session in sessions]
                )
                num_processed_tokens = numpy.stack(
                    [session.total_num_processed_tokens for session in sessions]
                )

                yield [
                    tokens,
                    logits,
                    finish_reason,
                    kv_cache_state,
                    num_processed_tokens,
                ]
            else:
                yield [tokens, logits, finish_reason]

    @staticmethod
    def causal_mask_input_present(model_path: str) -> bool:
        """
        Check whether the model has causal_mask input present or not.
        In general, the absence of causal_mask input means that the model
        cannot be run through the multitoken engine.

        :param model_path: path to the model
        :return: True if causal_mask input is present, False otherwise
        """
        is_causal_mask_input = any(
            inp.name == "causal_mask"
            for inp in onnx.load(model_path, load_external_data=False).graph.input
        )
        if not is_causal_mask_input:
            _LOGGER.warning(
                "This ONNX graph does not support processing the prompt"
                "with processing length > 1"
            )

        return is_causal_mask_input

    def get_kv_cache_decoder(self, engine_inputs: List[Any]) -> DecoderKVCache:
        """
        Initialize the kv cache decoder for the inference

        :return: the initialized kv cache decoder
        """
        engine = self.multitoken_engine or self.engine

        kv_cache_state = initialize_kv_cache_state(
            cache_shape=engine.cache_shape,
            kv_cache_data_type=engine.kv_cache_data_type,
            output_names=engine.output_names,
            length=self.sequence_length - self.prompt_sequence_length,
            empty=bool(self.internal_kv_cache),
        )

        kv_cache = DecoderKVCache(self.internal_kv_cache)
        kv_cache.setup(
            state=kv_cache_state,
            freeze_first_position=prepends_bos_token(self.tokenizer),
        )
        return kv_cache

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
