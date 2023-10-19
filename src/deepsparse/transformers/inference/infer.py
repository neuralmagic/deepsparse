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

"""
Usage: deepsparse.infer [OPTIONS] MODEL_PATH

  Command Line utility to interact with a text genration LLM in a chatbot
  style

  Example usage:

  deepsparse.infer [OPTIONS] <MODEL_PATH>

Options:
  --sequence_length INTEGER       Sequence length to compile model and
                                  tokenizer for.This controls the maximum
                                  context length of the pipeline.  [default:
                                  512]
  --sampling_temperature FLOAT    The temperature to use when samplingfrom the
                                  probability distribution computed from the
                                  logits.Higher values will result in more
                                  random samples. Shouldbe greater than 0.0.
                                  [default: 1.0]
  --prompt_sequence_length INTEGER
                                  Processed prompt in chunks of this length.
                                  This is to maximize the inference speed
                                  [default: 64]
  --show_tokens_per_sec / --no_show_tokens_per_sec
                                  Whether to display the token generation
                                  speed or not  [default:
                                  no_show_tokens_per_sec]
  --task TEXT                     The task to use for the pipeline. Choose any
                                  of `chat`, `codegen`, `text-generation`
                                  [default: chat]
  --stream / --no_stream          Whether to stream output as generated or not
                                  [default: no_stream]
  --help                          Show this message and exit.  [default:
                                  False]

Installation: pip install deepsparse[transformers]
Examples:

1) Use a local deployment directory
deepsparse.infer models/llama/deployment

2) Use a SparseZoo stub
deepsparse.infer \
    zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none # noqa: E501

3) Display token generation speed
deepsparse.infer models/llama/deployment \
    --show_tokens_per_sec

4) Disable history
deepsparse.infer models/llama/deployment \
    --task text-generation

5) Stream output
deepsparse.infer models/llama/deployment \
    --stream
"""

from typing import Optional

import click

from deepsparse import Pipeline
from deepsparse.tasks import SupportedTasks
from deepsparse.transformers.inference.prompt_parser import PromptParser


@click.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"), show_default=True
    )
)
@click.argument("model_path", type=str)
@click.option(
    "--data",
    type=str,
    default=None,
    help="Path to .txt, .csv, .json, or .jsonl file to load data from"
    "If provided, runs inference over the entire dataset. If not provided "
    "runs an interactive inference session in the console. Default None.",
)
@click.option(
    "--sequence_length",
    type=int,
    default=512,
    help="Sequence length to compile model and tokenizer for."
    "This controls the maximum context length of the pipeline.",
)
@click.option(
    "--sampling_temperature",
    type=float,
    default=1.0,
    help="The temperature to use when sampling"
    "from the probability distribution computed from the logits."
    "Higher values will result in more random samples. Should"
    "be greater than 0.0.",
)
@click.option(
    "--prompt_sequence_length",
    type=int,
    default=16,
    help="Processed prompt in chunks of this length. "
    "This is to maximize the inference speed",
)
@click.option(
    "--show_tokens_per_sec/--no_show_tokens_per_sec",
    default=False,
    help="Whether to display the token generation speed or not",
)
@click.option(
    "--task",
    default="text-generation",
    type=str,
    help="The task to use for the pipeline. Choose any of "
    "`chat`, `codegen`, `text-generation`",
)
@click.option(
    "--stream/--no_stream",
    is_flag=True,
    default=False,
    help="Whether to stream output as generated or not",
)
def main(
    model_path: str,
    data: Optional[str],
    sequence_length: int,
    sampling_temperature: float,
    prompt_sequence_length: int,
    show_tokens_per_sec: bool,
    task: str,
    stream: bool,
):
    """
    Command Line utility to interact with a text genration LLM in a chatbot style

    Example usage:

    deepsparse.infer [OPTIONS] <MODEL_PATH>
    """
    session_ids = "chatbot_cli_session"

    pipeline = Pipeline.create(
        task=task,  # let the pipeline determine if task is supported
        model_path=model_path,
        sequence_length=sequence_length,
        prompt_sequence_length=prompt_sequence_length,
    )

    if data:
        prompt_parser = PromptParser(data)
        default_prompt_kwargs = {
            "sequence_length": sequence_length,
            "sampling_temperature": sampling_temperature,
            "show_tokens_per_sec": show_tokens_per_sec,
        }

        for prompt_kwargs in prompt_parser.parse_as_iterable(**default_prompt_kwargs):
            _run_inference(
                task=task,
                pipeline=pipeline,
                session_ids=session_ids,
                stream=stream,
                **prompt_kwargs,
            )
        return

    # continue prompts until a keyboard interrupt
    while True:
        input_text = input("User: ")
        _run_inference(
            pipeline=pipeline,
            sampling_temperature=sampling_temperature,
            task=task,
            session_ids=session_ids,
            show_tokens_per_sec=show_tokens_per_sec,
            stream=stream,
            prompt=input_text,
        )


def _run_inference(
    pipeline: Pipeline,
    sampling_temperature: float,
    task: str,
    session_ids: str,
    show_tokens_per_sec: bool,
    prompt: str,
    stream: bool = False,
):
    pipeline_inputs = dict(
        prompt=[prompt],
        temperature=sampling_temperature,
    )

    if SupportedTasks.is_chat(task):
        pipeline_inputs["session_ids"] = session_ids

    response = pipeline(**pipeline_inputs, streaming=stream)
    _display_bot_response(stream, response)

    if show_tokens_per_sec:
        _display_generation_speed(prompt, pipeline)


def _display_generation_speed(prompt, pipeline):
    # display prefill and generation speed(s) in tokens/sec
    times = pipeline.timer_manager.times
    prefill_speed = (
        len(pipeline.tokenizer(prompt)["input_ids"]) / times["engine_prompt_prefill"]
    )
    generation_speed = 1.0 / times["engine_token_generation_single"]
    print(
        f"[prefill: {prefill_speed:.2f} tokens/sec]",
        f"[decode: {generation_speed:.2f} tokens/sec]",
        sep="\n",
    )


def _display_bot_response(stream: bool, response):
    # print response from pipeline, streaming or not

    print("Bot:", end="", flush=True)
    if stream:
        for generation in response:
            print(generation.generations[0].text, end="", flush=True)
        print()
    else:
        print(response.generations[0].text)


if "__main__" == __name__:
    main()
