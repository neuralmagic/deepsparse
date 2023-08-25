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

import argparse

from transformers import TextStreamer

from deepsparse import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate an interactive text-generation interface " "to evaluate a model."
        )
    )
    parser.add_argument(
        "model_path",
        type=str,
        help=(
            "Path to SparseZoo stub or model directory containing "
            "model.onnx, config.json, and tokenizer.json"
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Total sequence and context length",
    )
    args = parser.parse_args()

    # Construct pipelines
    ds_pipe = Pipeline.create(
        task="text-generation",
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        max_generated_tokens=args.max_new_tokens,
        prompt_processing_sequence_length=1,
        use_deepsparse_cache=False,
        engine_type="deepsparse",
        trust_remote_code=True,
    )
    ort_pipe = Pipeline.create(
        task="text-generation",
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        max_generated_tokens=args.max_new_tokens,
        prompt_processing_sequence_length=1,
        use_deepsparse_cache=False,
        engine_type="onnxruntime",
        trust_remote_code=True,
    )

    print("Welcome to the interactive text generation interface!")
    print("Type 'exit' to end the session.")

    while True:
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Ending the session. Goodbye!")
            break

        streamer = TextStreamer(ds_pipe.tokenizer)

        print("DeepSparse output:")
        _ = ds_pipe(sequences=user_input, streamer=streamer)

        print("ORT output:")
        _ = ort_pipe(sequences=user_input, streamer=streamer)


if __name__ == "__main__":
    main()
