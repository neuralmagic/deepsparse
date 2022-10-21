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
Script to faciliate benchmark sweeps over a variety of models, scenarios, and inputs

##########
Example sweep over BERT models:
    models = [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni",
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant_3layers-aggressive_84",
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
"""

import csv
import time
from itertools import product
from typing import Dict, List

from deepsparse.benchmark.benchmark_model import benchmark_model


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
                "Input Shape",
                "Scenario",
                "Num Streams",
                "Throughput",
                "Mean Latency",
                "Median Latency",
                "Std Latency",
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
            for num_streams in num_streams_list:
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
                        batch_size,
                        input_shape,
                        scenario,
                        num_streams,
                        items_per_second,
                        latency_mean,
                        latency_median,
                        latency_std,
                        command_str,
                    ]
                )
