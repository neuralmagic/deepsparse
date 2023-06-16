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
Example use:
python src/deepsparse/transformers/eval_text_generation.py 
--task opt 
--model_path /home/ubuntu/damian/sparseml/deployment 
--sentence "While the Indiana Jones movies likely inspired a lot of young viewers"
--sequence_length 128 
"""  # noqa: W291
import argparse
from typing import Any, Dict

from deepsparse import Pipeline
from deepsparse.transformers.metrics import Perplexity


def perplexity_eval(args: argparse.Namespace) -> Dict[str, Any]:

    pipeline = Pipeline.create(
        task=args.task,
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        engine_type=args.engine_type,
        max_generated_tokens=1,
    )

    perplexity = Perplexity(pipeline=pipeline)
    perplexity.add_batch(args.sentence)

    return perplexity.compute()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate perplexity of a text-generation model on a dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        help="A sentence to evaluate perplexity on",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        help="Sequence length of the pipeline to use for evaluation",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to use for evaluation",
    )
    parser.add_argument(
        "--engine_type",
        type=str,
        default="onnxruntime",
        help="Engine type to use for evaluation",
    )

    args = parser.parse_args()
    results = perplexity_eval(args)
    print(results)
