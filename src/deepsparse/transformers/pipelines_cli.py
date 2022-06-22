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
deepsparse.transformers.run_inference entrypoint cli script for one shot
inference using pipelines

####################
usage: deepsparse.transformers.run_inference [-h]
                           [-t {ner,question-answering,sentiment-analysis,
                           text-classification,token-classification}]
                           -d DATA --model-path
                           MODEL_PATH [--engine-type {deepsparse,onnxruntime}]
                           [--max-length MAX_LENGTH] [--num-cores NUM_CORES]
                           [-b BATCH_SIZE] [--scheduler {multi,single}]
                           [-o OUTPUT_FILE]

Cli utility for one shot inference using pipelines

optional arguments:
  -h, --help            show this help message and exit
  -t {ner,question-answering,sentiment-analysis,text-classification,
  token-classification},
  --task {ner,question-answering,sentiment-analysis,text-classification,
  token-classification}
                        Name of the task to define which pipeline to
                        create.Currently supported tasks ['ner',
                        'question-answering', 'sentiment-analysis', 'text-
                        classification', 'token-classification']
  -d DATA, --data DATA  Path to file containing data for inferences, inputs
                        should be separated via newline
  --model-path MODEL_PATH, --model_path MODEL_PATH
                        Path to (ONNX) model file to run, can also be a
                        SparseZoo stub
  --engine-type {deepsparse,onnxruntime}, --engine_type {deepsparse,onnxruntime}
                        inference engine name to use. Supported options are
                        'deepsparse'and 'onnxruntime'
  --max-length MAX_LENGTH, --max_length MAX_LENGTH
                        Maximum sequence length of model inputs. default is
                        128
  --num-cores NUM_CORES, --num_cores NUM_CORES
                        Number of CPU cores to run engine with. Default is the
                        maximum available
  -b BATCH_SIZE, --batch-size BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to use for pipelines
  --scheduler {multi,single}
                        The scheduler to use for the engine. Can be None,
                        single or multi
  -o OUTPUT_FILE, --output-file OUTPUT_FILE, --output_file OUTPUT_FILE
                        Directs the output to a name of your choice
####################
Example commands:
1) deepsparse.transformers.run_inference --task ner \
    --model-path bert-ner-test.onnx \
    --data input.txt

2) deepsparse.transformers.run_inference --task ner \
    --model-path models/bert-ner-test.onnx \
    --data input.txt \
    --output-file out.txt \
    --batch_size 2

3) deepsparse.transformers.run_inference --task sentiment-analysis \
    --model_path models/bert-sst-test.onnx \
    --data models/input.txt \
    --batch_size 2 \
    --output_file out.txt

"""

import argparse
import json
from typing import Any, Callable

from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.pipeline import SUPPORTED_PIPELINE_ENGINES
from deepsparse.transformers import fix_numpy_types
from deepsparse.transformers.loaders import SUPPORTED_EXTENSIONS, get_batch_loader


__all__ = [
    "cli",
]

SUPPORTED_TASKS = [
    "question_answering",
    "text_classification",
    "token_classification",
    "ner",
    "zero_shot_text_classification",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cli utility for one shot inference using pipelines"
    )

    parser.add_argument(
        "-t",
        "--task",
        help="Name of the task to define which pipeline to create."
        f" Currently supported tasks {SUPPORTED_TASKS}",
        choices=SUPPORTED_TASKS,
        type=str,
        default="sentiment-analysis",
    )

    parser.add_argument(
        "-d",
        "--data",
        help="Path to file containing data for inferences, "
        f"inputs should be separated via newline, Supports "
        f"{SUPPORTED_EXTENSIONS} files",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--model-path",
        "--model_path",
        help="Path to model directory containing `model.onnx`, `config.json`,"
        "and `tokenizer.json` files, ONNX model file, or SparseZoo stub",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--engine-type",
        "--engine_type",
        help="Inference engine name to use. Supported options are 'deepsparse'"
        "and 'onnxruntime'",
        type=str,
        choices=SUPPORTED_PIPELINE_ENGINES,
        default=SUPPORTED_PIPELINE_ENGINES[0],
    )

    parser.add_argument(
        "--max-length",
        "--max_length",
        help="Maximum sequence length of model inputs. default is 128",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-cores",
        "--num_cores",
        help="Number of CPU cores to run engine with. Default is the maximum "
        "available",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        "--batch_size",
        help="The batch size to use for pipelines",
        type=int,
        default=1,
    )

    scheduler_choices = ["multi", "single"]
    parser.add_argument(
        "--scheduler",
        choices=scheduler_choices,
        help="The scheduler to use for the engine. Can be None, single or multi",
        default=scheduler_choices[0],
    )

    parser.add_argument(
        "-o",
        "--output-file",
        "--output_file",
        action="store",
        dest="output_file",
        type=str,
        help="Directs json output to a name of your choice, defaults to " "out.json",
        default="out.json",
    )

    _args = parser.parse_args()
    return _args


def cli():
    """
    Cli entrypoint for one-shot inference using pipelines
    """
    _args = _parse_args()

    pipe = Pipeline.create(
        task=_args.task,
        model_path=_args.model_path,
        engine_type=_args.engine_type,
        sequence_length=_args.max_length,
        num_cores=_args.num_cores,
        batch_size=_args.batch_size,
        scheduler=_args.scheduler,
    )
    process_dataset(
        pipeline_object=pipe,
        data_path=_args.data,
        output_path=_args.output_file,
        batch_size=_args.batch_size,
        task=_args.task,
    )


def response_to_json(response: Any):
    """
    Converts a response to a json string

    :param response: A List[Any] or Dict[Any, Any] or a Pydantic model,
        that should be converted to a valid json string
    :return: A json string representation of the response
    """
    if isinstance(response, list):
        return [response_to_json(val) for val in response]
    elif isinstance(response, dict):
        return {key: response_to_json(val) for key, val in response.items()}
    elif isinstance(response, BaseModel):
        return response.dict()
    return json.dumps(response)


def process_dataset(
    pipeline_object: Callable,
    data_path: str,
    batch_size: int,
    task: str,
    output_path: str,
) -> None:
    """
    :param pipeline_object: An instantiated pipeline Callable object
    :param data_path: Path to input file, supports csv, json and text files
    :param batch_size: batch_size to use for inference
    :param task: The task pipeline is instantiated for
    :param output_path: Path to a json file to output inference results to
    """
    batch_loader = get_batch_loader(
        data_file=data_path,
        batch_size=batch_size,
        task=task,
    )
    # Wraps pipeline object to make numpy types serializable
    pipeline_object = fix_numpy_types(pipeline_object)
    with open(output_path, "a") as output_file:
        for batch in batch_loader:
            if task == "question_answering":
                # Un-Wrap list cause only batch_size 1 is supported
                batch = {key: value[0] for key, value in batch.items()}
            batch_output = pipeline_object(**batch)
            json_output = response_to_json(batch_output)
            json.dump(json_output, output_file)
            output_file.write("\n")


if __name__ == "__main__":
    cli()
