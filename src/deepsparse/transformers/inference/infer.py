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
  --help                          Show this message and exit.

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
    default=64,
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
    default="chat",
    type=str,
    help="The task to use for the pipeline. Choose any of "
    "`chat`, `codegen`, `text-generation`",
)
def main(
    model_path: str,
    data: Optional[str],
    sequence_length: int,
    sampling_temperature: float,
    prompt_sequence_length: int,
    show_tokens_per_sec: bool,
    task: str,
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
            "prompt_sequence_length": prompt_sequence_length,
            "show_tokens_per_sec": show_tokens_per_sec,
        }

        for prompt_kwargs in prompt_parser.parse_as_iterable(**default_prompt_kwargs):
            _run_inference(
                task=task,
                pipeline=pipeline,
                session_ids=session_ids,
                **prompt_kwargs,
            )
        return

    # continue prompts until a keyboard interrupt
    while data is None:  # always True in interactive Mode
        prompt = input(">>> ")
        _run_inference(
            pipeline,
            sampling_temperature,
            task,
            session_ids,
            show_tokens_per_sec,
            prompt_sequence_length,
            prompt,
        )


def _run_inference(
    pipeline,
    sampling_temperature,
    task,
    session_ids,
    show_tokens_per_sec,
    prompt_sequence_length,
    prompt,
    **kwargs,
):
    pipeline_inputs = dict(
        prompt=[prompt],
        temperature=sampling_temperature,
        **kwargs,
    )
    if SupportedTasks.is_chat(task):
        pipeline_inputs["session_ids"] = session_ids

    response = pipeline(**pipeline_inputs)
    print("\n", response.generations[0].text)

    if show_tokens_per_sec:
        times = pipeline.timer_manager.times
        prefill_speed = (
            1.0 * prompt_sequence_length / times["engine_prompt_prefill_single"]
        )
        generation_speed = 1.0 / times["engine_token_generation_single"]
        print(
            f"[prefill: {prefill_speed:.2f} tokens/sec]",
            f"[decode: {generation_speed:.2f} tokens/sec]",
            sep="\n",
        )


if __name__ == "__main__":
    main()
