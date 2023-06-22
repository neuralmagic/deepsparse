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
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer

from deepsparse import Pipeline
from deepsparse.transformers.engines import NLDecoderEngine
from deepsparse.transformers.helpers import get_onnx_path_and_configs
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = ["TextGenerationPipeline"]


class TextGenerationInput(BaseModel):
    sequence: str = Field(
        description="The input sequence to generate the text from.",
    )
    return_logits: bool = Field(
        default=False,
        description="A flag that indicates whether to return "
        "the logits for the generated text sequence. ",
    )


class TextGenerationOutput(BaseModel):
    sequence: str = Field(
        description="The generated text sequence.",
    )
    logits: Optional[numpy.ndarray] = Field(
        default=None,
        description="The logits for the generated text sequence."
        "The logits have dimensions "
        "[batch_size, sequence_length, vocab_size]",
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
        Higher values will result in more random samples.
    :param max_generated_tokens: the maximum number of tokens to generate
        given the input sequence. If None, the model will generate
        tokens until the end of the sequence is reached.
        Otherwise, it will generate up to the maximum number of tokens or end of
        sequence is reached.
    :param prompt_batch_threshold: the threshold for the ratio of running the prompt
        as a single inference vs running the prompt auto-regressively.
        If the number of input sequences divided by the max sequence length is
        greater than the threshold, the prompt will be run as a single inference.
        Default is None, which will always run auto-regressively.
    :param force_max_tokens: if True, the pipeline will generate the maximum number
        of tokens supplied even if the stop token is reached.
    :param kwargs: kwargs to pass to the TransformersPipeline
    """

    def __init__(
        self,
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        max_generated_tokens: Optional[int] = 4096,
        prompt_batch_threshold: Optional[float] = None,
        force_max_tokens: bool = False,
        **kwargs,
    ):
        if kwargs["engine_type"] == "deepsparse":
            raise NotImplementedError(
                "The text generation pipeline is not "
                "supported for the deepsparse engine"
            )

        super().__init__(**kwargs, _delay_engine_initialize=True)

        if self._batch_size != 1:
            raise ValueError("Only batch size 1 is supported for generation pipelines")

        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.max_generated_tokens = max_generated_tokens
        self.prompt_batch_threshold = prompt_batch_threshold
        self.force_max_tokens = force_max_tokens

        # override tokenizer to pad to left
        self.tokenizer.padding_side = "left"
        self.engine = NLDecoderEngine(
            onnx_file_path=self.onnx_file_path,
            multitoken=False,
            engine_type=self.engine_type,
            engine_args=self.engine_args,
            engine_context=self.context,
            sampling_temperature=self.sampling_temperature,
            deterministic=self.deterministic,
            sequence_length=self.sequence_length,
        )
        self.multitoken_engine = None
        if prompt_batch_threshold is not None:
            self.multitoken_engine = NLDecoderEngine(
                onnx_file_path=self.onnx_file_path,
                multitoken=True,
                engine_type=self.engine_type,
                engine_args=self.engine_args,
                context=self.context,
                sampling_temperature=self.sampling_temperature,
                deterministic=self.deterministic,
                sequence_length=self.sequence_length,
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

        attention_mask = input_tokens["attention_mask"]

        # TODO: Positions input is not required by BLOOM
        # let's make it optional in the future
        positions = attention_mask.cumsum(1) * attention_mask
        positions -= 1  # assert that positions start at 0
        positions_input = dict(positions=positions)

        input_tokens = {**input_tokens, **positions_input}
        onnx_input_names = self.engine.onnx_input_names_no_cache
        engine_input = self.tokens_to_engine_input(input_tokens, onnx_input_names)

        return engine_input

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> TextGenerationOutput:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :return: the output schema for the pipeline
        """
        generated_tokens, generated_logits = engine_outputs
        # TODO: Make sure it works with batch size > 1
        sequence = self.tokenizer.decode(
            generated_tokens[0][0], skip_special_tokens=True
        )
        logits = generated_logits if kwargs.get("return_logits") else None

        return TextGenerationOutput(sequence=sequence, logits=logits)

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
        # run the prompt through
        tokens, logits = self.prompt_inference(engine_inputs)
        num_prompt_tokens = len(tokens) - 1

        # create the generated output
        max_tokens = (
            self.max_generated_tokens
            if self.max_generated_tokens and self.max_generated_tokens > 0
            else 100 * self.sequence_length
        )  # set safety for absolute max generation

        generated_tokens = [tokens[-1]]
        generated_logits = [logits]

        while len(generated_tokens) < max_tokens:
            (
                token,
                logits,
            ) = self.autoregressive_inference(tokens, num_prompt_tokens)
            tokens.append(token)
            generated_tokens.append(token)
            generated_logits.append(logits)

            if token == self.tokenizer.eos_token_id and not self.force_max_tokens:
                break

        return numpy.array([[generated_tokens]]), numpy.concatenate(
            generated_logits, axis=1
        )

    def prompt_inference(
        self, engine_inputs: List[numpy.ndarray]
    ) -> Tuple[List[int], Dict[str, numpy.ndarray]]:
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
        new_token = None

        if (
            self.prompt_batch_threshold is None
            or self.prompt_batch_threshold >= 1
            or len(tokens) / float(self.sequence_length) < self.prompt_batch_threshold
        ):
            # prompt size is small, run autoregressive inference to populate kv cache
            run_tokens = []
            for token in tokens:
                run_tokens.append(token)
                new_token, new_logits = self.autoregressive_inference(
                    run_tokens, num_prompt_tokens=0
                )
        else:
            # larger prompt size, run through multi-token engine in single pass
            new_token, new_logits = self.multitoken_engine(engine_inputs)

        tokens.append(new_token)

        return tokens, new_logits

    def autoregressive_inference(
        self,
        tokens: List[int],
        num_prompt_tokens: int,
    ) -> Tuple[int, numpy.ndarray]:
        """
        An inference run that processes the last token to generate
        a new token and new logits.

        :param tokens: The current context (prompt + generated tokens so far)
        :param num_prompt_tokens: the number of tokens in the initial prompt
        :return: The new, generated token and the logits for the new token
            (with dimensions ['batch_size', 'num_tokens', 'vocab_size'])
        """
        new_token = tokens[-1]
        # padding is added to left, so attention mask is 1s from the
        # right up to the number of total tokens (prompt + generated)
        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        num_tokens_processed = min(len(tokens), self.sequence_length)  # cap by seq len

        input_ids = numpy.array([[new_token]])
        engine_inputs = [input_ids, attention_mask]

        generated_token, generated_logits = self.engine(engine_inputs)

        return generated_token, generated_logits

    # TODO: Let's discuss whether we need this, maybe we can
    # simplify more
    def setup_onnx_file_path(self) -> str:
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
            self.model_path, model_max_length=self.sequence_length
        )
        self.config_path = os.path.join(config_path, "config.json")

        return onnx_path
