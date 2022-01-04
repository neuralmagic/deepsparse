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
deepsparse.pipelines entrypoint cli script for one shot inference using
pipelines

####################
usage: deepsparse.pipeline [-h]
                           [-t {ner,question-answering,sentiment-analysis,
                           text-classification,token-classification}]
                           -d DATA [--model-name MODEL_NAME] --model-path
                           MODEL_PATH [--engine-type {deepsparse,onnxruntime}]
                           [--config CONFIG] [--tokenizer TOKENIZER]
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
                        create.Currently supported tasks dict_keys(['ner',
                        'question-answering', 'sentiment-analysis', 'text-
                        classification', 'token-classification'])
  -d DATA, --data DATA  Path to file containing data for inferences, inputs
                        should be separated via newline
  --model-name MODEL_NAME, --model_name MODEL_NAME
                        Canonical name of the hugging face model this model is
                        based on
  --model-path MODEL_PATH, --model_path MODEL_PATH
                        path to (ONNX) model file to run
  --engine-type {deepsparse,onnxruntime}, --engine_type {deepsparse,onnxruntime}
                        inference engine name to use. Supported options are
                        'deepsparse'and 'onnxruntime'
  --config CONFIG       huggingface model config, if none provided, default
                        will be usedwhich will be from the model name or
                        sparsezoo stub if given for model path
  --tokenizer TOKENIZER
                        huggingface tokenizer, if none provided, default will
                        be used
  --max-length MAX_LENGTH, --max_length MAX_LENGTH
                        maximum sequence length of model inputs. default is
                        128
  --num-cores NUM_CORES, --num_cores NUM_CORES
                        number of CPU cores to run engine with. Default is the
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
1) deepsparse.pipeline --task ner \
    --model-path bert-ner-test.onnx \
    --data input.txt

2) deepsparse.pipeline --task ner \
    --model-path models/bert-ner-test.onnx \
    --data input.txt \
    --config ner-config.json \
    --output-file out.txt \
    --batch_size 2

3) deepsparse.pipeline --task sentiment-analysis \
    --model_path models/bert-sst-test.onnx \
    --data models/input.txt \
    --batch_size 2 \
    --output_file out.txt

"""

import argparse
import json
from csv import reader
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from transformers import PretrainedConfig, PreTrainedTokenizer

from .pipelines import SUPPORTED_ENGINES, SUPPORTED_TASKS, pipeline


__all__ = [
    "cli",
]


@dataclass
class _PipelineCliArgs:
    """
    :param task: name of the task to define which pipeline to create. Currently
        supported task - ['ner', 'question-answering', 'sentiment-analysis',
        'text-classification', 'token-classification']
    :param data: str, List[str], file containing List of strings representing input
    :param model_name: canonical name of the hugging face model this model is based on
    :param model_path: path to (ONNX) model file to run
    :param engine_type: inference engine name to use. Supported options are 'deepsparse'
        and 'onnxruntime'
    :param config: huggingface model config, if none provided, default will be used
        which will be from the model name or sparsezoo stub if given for model path
    :param tokenizer: huggingface tokenizer, if none provided, default will be used
    :param max_length: maximum sequence length of model inputs. default is 128
    :param num_cores: number of CPU cores to run engine with. Default is the maximum
        available
    :param batch_size: The batch size to use for pipelines. Defaults to 1.
    :param scheduler: The scheduler to use for the engine. Can be None, single or multi.
    :param output_file: File to write output to, omit to print output to stdout
    :return: Inference results  for each input
    """

    task: str
    data: Union[List, str]
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    engine_type: str = SUPPORTED_ENGINES[0]
    config: Optional[Union[str, PretrainedConfig]] = None
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None
    max_length: int = 128
    num_cores: Optional[int] = None
    batch_size: int = 1
    scheduler: Optional[str] = None
    output_file: Optional[argparse.FileType] = None

    def __post_init__(self):
        data_file = Path(self.data)
        assert data_file.exists(), f"The input data file must exist, {data_file}"
        assert data_file.is_file(), f"The input data file must be a file, {data_file}"
        supported_extensions = [".json", ".csv", ".txt"]
        assert data_file.suffix in supported_extensions, (
            f"The input data file must be one of {supported_extensions}, "
            f"found {data_file.suffix}"
        )

    @property
    def data_loader(self):
        data_file = Path(self.data)
        extension = data_file.suffix
        with open(data_file) as f:
            if extension == ".json":
                data = json.load(f)
                gen = data.values()

            elif extension == ".csv":
                gen = reader(f)

            elif extension == ".txt":
                gen = (line.strip() for line in f)

            for _input in gen:
                yield _input

    @property
    def batch_loader(self):
        batch = []

        for _ in self.data_loader:
            batch.append(_)
            if len(batch) == self.batch_size:
                yield batch
                batch = []


def _parse_args() -> _PipelineCliArgs:
    parser = argparse.ArgumentParser(
        description="Cli utility for one shot inference using pipelines"
    )

    parser.add_argument(
        "-t",
        "--task",
        help="Name of the task to define which pipeline to create."
        f"Currently supported tasks {SUPPORTED_TASKS.keys()}",
        choices=SUPPORTED_TASKS.keys(),
        type=str,
        default="sentiment-analysis",
    )

    parser.add_argument(
        "-d",
        "--data",
        help="Path to file containing data for inferences, "
        "inputs should be separated via newline",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--model-name",
        "--model_name",
        help="Canonical name of the hugging face model this model is based on",
        default=None,
        type=Optional[str],
    )

    parser.add_argument(
        "--model-path",
        "--model_path",
        help="path to (ONNX) model file to run",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--engine-type",
        "--engine_type",
        help="inference engine name to use. Supported options are 'deepsparse'"
        "and 'onnxruntime'",
        type=str,
        choices=SUPPORTED_ENGINES,
        default=SUPPORTED_ENGINES[0],
    )

    parser.add_argument(
        "--config",
        help="huggingface model config, if none provided, default will be used"
        "which will be from the model name or sparsezoo stub if given for "
        "model path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tokenizer",
        help="huggingface tokenizer, if none provided, default will be used",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-length",
        "--max_length",
        help="maximum sequence length of model inputs. default is 128",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-cores",
        "--num_cores",
        help="number of CPU cores to run engine with. Default is the maximum "
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
        type=argparse.FileType("w"),
        help="Directs the output to a name of your choice",
        default="-",
    )

    _args = parser.parse_args()
    return _PipelineCliArgs(**vars(_args))


def cli():
    """
    Cli entrypoint for one-shot inference using pipelines
    """
    _args = _parse_args()

    pipe = pipeline(
        task=_args.task,
        model_name=_args.model_name,
        model_path=_args.model_path,
        engine_type=_args.engine_type,
        config=_args.config,
        tokenizer=_args.tokenizer,
        max_length=_args.max_length,
        num_cores=_args.num_cores,
        batch_size=_args.batch_size,
        scheduler=_args.scheduler,
    )

    for batch in _args.batch_loader:
        batch_output = pipe(batch)
        _args.output_file.write(str(batch_output))


if __name__ == "__main__":
    cli()
