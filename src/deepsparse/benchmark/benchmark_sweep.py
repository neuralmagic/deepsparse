#!/usr/bin/env python

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
Usage: benchmark_sweep.py [OPTIONS]

  Script to run benchmark sweep over a directory containing models or a comma
  separated list of model-paths/Sparsezoo stubs against different
  `batch_sizes` and `num_cores` for ORT and deepsparse run-times, writes one
  csv per model file

  Examples:

      1) Benchmark sweep over model-paths for both ORT and DeepSparse for
      multiple `num-cores` and `batch-sizes`:

          python benchmark_sweep.py --model-paths
          "~/models/resnet50.onnx, ~/quant-models/resnet_channel20_quant.onnx"
          --num-cores "1, 4, max" --batch-sizes "1, 16, 32"

Options:
  --num-cores TEXT       Comma separated values for different num_cores to
                         benchmark against, can also be specified as max to
                         benchmark against max num of cores
  --batch-sizes TEXT     Comma separated values for different batch_sizes to
                         benchmark against
  --model_paths TEXT     Comma separated list of model paths, or Sparsezoo
                         stubs, or directories containing onnx models
  --save-dir TEXT        Directory to save model benchmarks in  [default:
                         benchmarking-results]
  --run-time INTEGER     The run_time to execute model for  [default: 30]
  --warmup-time INTEGER  The warmup_time to execute model for  [default: 5]
  -e, --engine TEXT      Inference engine backend to run eval on. Choices are
                         'deepsparse', 'onnxruntime'. Default is 'deepsparse'
                         and 'onnxruntime'. Can also specify a user defined
                         engine class by giving the script and class name in
                         the following format <path to python script>:<Engine
                         Class name>. This engine class will be dynamically
                         imported during runtime. Note multiple engines can
                         also be specified
  --help                 Show this message and exit.
"""


import csv
import logging
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from deepsparse.benchmark.benchmark_model import benchmark_model
from deepsparse.cpu import cpu_details


_LOGGER = logging.getLogger(__name__)

__all__ = ["benchmark_sweep"]


def benchmark_sweep(
    models: List[str],
    batch_sizes: List[int] = [1],
    num_cores: List[int] = [None],
    scenario_streams_dict: Dict = {"sync": [None]},
    input_shapes: List[str] = [None],
    engines: List[str] = ["deepsparse"],
    run_time: int = 30,
    warmup_time: int = 5,
    export_csv_path: str = None,
):
    """
    Function to facilitate benchmark sweeps over a variety of models, scenarios,
    and inputs

    ##########
    Example sweep over BERT models:
        models = [
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad
            /12layer_pruned80_quant-none-vnni",
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad
            /pruned_quant_3layers-aggressive_84",
        ]
        batch_sizes = [1, 16]
        input_shapes = ["[1,128]", "[1,384]"]
        scenario_streams_dict = {
            "sync": [None],
            "async": [None, 2, 4],
        }
        engines = ["deepsparse", "onnxruntime"]

        benchmark_sweep(
            models=models,
            batch_sizes=batch_sizes,
            input_shapes=input_shapes,
            scenario_streams_dict=scenario_streams_dict,
            engines=engines,
        )

    :param models: a list of string model paths or SparseZoo stubs to benchmark
    :param batch_sizes: list of different batch sizes to benchmark against
    :param num_cores: list of different num_cores to benchmark against
    :param scenario_streams_dict: a dict containing different scenarios (sync, async)
        as keys and the corresponding num_streams to use as their values
    :param input_shapes: optional input shapes to use for benchmarking, if not specified
        then they are inferred from the onnx graph
    :param engines: list of strings representing different runtime engines to
        benchmark with, currently supported ones include `onnxruntime` and
        `deepsparse`
    :param run_time: number of seconds to execute each model for
    :param warmup_time: number of seconds to warmup each model for before benchmarking
    :param export_csv_path: path to the csv to write results to
    """

    if not export_csv_path:
        export_csv_path = "benchmark_sweep_{}.csv".format(
            time.strftime("%Y%m%d_%H%M%S")
        )

    print(f"Starting benchmarking sweep, writing result to {export_csv_path}")

    with open(export_csv_path, "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(
            [
                "Model",
                "Engine",
                "Batch Size",
                "Num Cores",
                "Scenario",
                "Num Streams",
                "Items per second",
                "Mean Latency",
                "Median Latency",
                "Std Latency",
                "Input Shape",
                "Command",
            ]
        )

        for (
            model,
            batch_size,
            num_core,
            input_shape,
            engine,
            (scenario, num_streams_list),
        ) in product(
            models,
            batch_sizes,
            num_cores,
            input_shapes,
            engines,
            scenario_streams_dict.items(),
        ):
            if num_cores is None:
                # override to max available num_cores if not specified
                num_cores = cpu_details()[0]

            for num_streams in num_streams_list:
                try:

                    result = benchmark_model(
                        model_path=model,
                        batch_size=batch_size,
                        input_shapes=input_shape,
                        num_cores=num_core,
                        scenario=scenario,
                        time=run_time,
                        warmup_time=warmup_time,
                        num_streams=num_streams,
                        engine=engine,
                    )

                    items_per_second = result["benchmark_result"]["items_per_sec"]
                    latency_mean = result["benchmark_result"]["mean"]
                    latency_median = result["benchmark_result"]["median"]
                    latency_std = result["benchmark_result"]["std"]

                    command = [
                        "deepsparse.benchmark",
                        f"{model}",
                        f"--batch_size={batch_size}",
                        f"--input_shapes={input_shape}",
                        f"--num_cores={num_core}",
                        f"--scenario={scenario}",
                        f"--time={run_time}",
                        f"--warmup_time={warmup_time}",
                        f"--num_streams={num_streams}",
                        f"--engine={engine}",
                    ]
                    command_str = f"\"{' '.join(command)}\""

                    writer.writerow(
                        [
                            model,
                            engine,
                            batch_size,
                            num_core,
                            scenario,
                            num_streams,
                            items_per_second,
                            latency_mean,
                            latency_median,
                            latency_std,
                            input_shape,
                            command_str,
                        ]
                    )
                    _LOGGER.info(
                        f"{model} benchmarking results written to {export_csv_path}"
                    )
                except Exception as exception:
                    _LOGGER.info(
                        f"An exception was raised while trying to benchmark {model}, "
                        f"with batch_size: {batch_size}, input shapes: {input_shape},"
                        f"num_cores: {num_core} and num_streams: {num_streams}, "
                        f"Exception: {exception}"
                    )


def _get_models(
    model_paths: Optional[str] = None,
) -> List[str]:
    models = []
    if not model_paths:
        raise ValueError(f"Expected valid model_paths but got {model_paths}")

    all_model_paths = model_paths.split(",")
    for model in all_model_paths:
        model = model.strip()

        if Path(model).is_file() or model.startswith("zoo:"):
            models.append(model)
        elif Path(model).is_dir():
            models.extend([str(model) for model in Path.rglob("*.onnx")])
        else:
            raise ValueError(
                "The specified models must either be valid paths that exist,"
                f"or a valid SparseZoo stub but found {model}"
            )
    if not models:
        raise ValueError(
            "Could not find any models, either `--model-dir` did not have `onnx` files "
            "or was not specified, additionally `--model-paths` was also not specified"
        )
    return models


@click.command()
@click.option(
    "--num-cores",
    help="Comma separated values for different num_cores to benchmark against, "
    "can also be specified as max to benchmark against max num of cores",
)
@click.option(
    "--batch-sizes",
    help="Comma separated values for different batch_sizes to benchmark against",
)
@click.option(
    "--model_paths",
    type=str,
    default=None,
    help="Comma separated list of model paths, or Sparsezoo stubs, or "
    "directories containing onnx models",
)
@click.option(
    "--save-dir",
    help="Directory to save model benchmarks in",
    default="benchmarking-results",
    show_default=True,
)
@click.option(
    "--run-time",
    type=int,
    default=30,
    help="The run_time to execute model for",
    show_default=True,
)
@click.option(
    "--warmup-time",
    type=int,
    default=5,
    help="The warmup_time to execute model for",
    show_default=True,
)
@click.option(
    "--engine",
    "-e",
    multiple=True,
    default=["onnxruntime", "deepsparse"],
    help="Inference engine backend to run eval on. Choices are 'deepsparse', "
    "'onnxruntime'. Default is 'deepsparse' and 'onnxruntime'. Can also specify a user "
    "defined engine class by giving the script and class name in the following format "
    "<path to python script>:<Engine Class name>. This engine class will be "
    "dynamically imported during runtime. Note multiple engines can also be specified",
)
def main(
    num_cores: str,
    batch_sizes: str,
    model_paths: Optional[str],
    save_dir: str,
    run_time: int,
    warmup_time: int,
    engine: List[str],
):
    """
    Script to run benchmark sweep over a directory containing models or a
    comma separated list of model-paths/Sparsezoo stubs against different `batch_sizes`
    and `num_cores` for ORT and deepsparse run-times, writes one csv per model file

    Examples:

        1) Benchmark sweep over model-paths for both ORT and DeepSparse for
        multiple `num-cores` and `batch-sizes`:

            python benchmark_sweep.py --model-paths \
            "~/models/resnet50.onnx, ~/quant-models/resnet_channel20_quant.onnx" \
                --num-cores "1, 4, max" --batch-sizes "1, 16, 32"
    """
    num_cores: List[int] = _validate_num_cores(num_cores)
    batch_sizes: List[int] = _validate_batch_sizes(batch_sizes)
    models: List[str] = _get_models(model_paths=model_paths)

    save_dir_path: Path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        if model.startswith("zoo:"):
            model_name = ""
        else:
            model_name = Path(model).name

        export_csv_name = f'benchmark_{model_name}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        export_csv_path = save_dir_path / export_csv_name
        benchmark_sweep(
            models=[str(model)],
            batch_sizes=batch_sizes,
            num_cores=num_cores,
            engines=engine,
            scenario_streams_dict={
                "sync": [None],
            },
            warmup_time=warmup_time,
            run_time=run_time,
            export_csv_path=str(export_csv_path),
        )


def _remove_duplicates(items: List[Any]):
    return list(set(items))


def _validate_batch_sizes(batch_sizes: str):
    batch_sizes = [int(batch_size.strip()) for batch_size in batch_sizes.split(",")]
    return _remove_duplicates(items=batch_sizes)


def _validate_num_cores(num_cores: str):
    valid_num_cores = []
    for cores in num_cores.split(","):
        cores = cores.strip()
        if cores == "max":
            new_core_value = cpu_details()[0]
        else:
            new_core_value = int(cores.strip())
        valid_num_cores.append(new_core_value)
    return _remove_duplicates(items=valid_num_cores)


if __name__ == "__main__":
    main()
