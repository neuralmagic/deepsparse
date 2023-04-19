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
import logging
from typing import Any, Dict, List, Optional

import click
import onnx
from onnx import ModelProto

from deepsparse import model_debug_analysis
from deepsparse.benchmark.benchmark_model import benchmark_model
from deepsparse.utils import generate_random_inputs, model_to_path
from sparsezoo import convert_to_bool
from sparsezoo.analyze import (
    BenchmarkResult,
    BenchmarkScenario,
    ImposedSparsificationInfo,
    ModelAnalysis,
    NodeInferenceResult,
)
from sparsezoo.analyze.cli import analyze_options, analyze_performance_options


_LOGGER = logging.getLogger(__name__)


@click.command()
@analyze_options
@analyze_performance_options
def main(
    model_path: str,
    save: str,
    batch_size_throughput: int,
    benchmark_engine: str,
    by_layers: Optional[str],
    by_types: Optional[str],
    compare: Optional[str],
    **kwargs,
):
    """
    DeepSparse Performance analysis for ONNX models.

    MODEL_PATH: can be a SparseZoo stub, or local path to a
    deployment-directory or ONNX model

    Examples:

    - Run model analysis on resnet

        deepsparse.analyze ~/models/resnet50.onnx
    """
    logging.basicConfig(level=logging.INFO)

    for unimplemented_feat in (
        "save_graphs",
        "impose",
    ):
        if kwargs.get(unimplemented_feat):
            raise NotImplementedError(
                f"--{unimplemented_feat} has not been implemented yet"
            )

    _LOGGER.info("Starting Analysis ...")
    analysis = ModelAnalysis.create(model_path)
    _LOGGER.info("Analysis complete, collating results...")
    scenario = BenchmarkScenario(
        batch_size=batch_size_throughput,
        num_cores=None,
        engine=benchmark_engine,
    )
    performance_summary = run_benchmark_and_analysis(
        onnx_model=model_to_path(model_path),
        scenario=scenario,
    )
    by_types: bool = convert_to_bool(by_types)
    by_layers: bool = convert_to_bool(by_layers)

    analysis.benchmark_results = [performance_summary]
    summary = analysis.summary(
        by_types=by_types,
        by_layers=by_layers,
    )
    summary.pretty_print()

    if compare is not None:
        if "," in compare:
            compare = compare.split(",")
        else:
            compare = [compare]

        print("Comparison Analysis:")
        for model_to_compare in compare:
            compare_model_analysis = ModelAnalysis.create(model_to_compare)
            _LOGGER.info(f"Running Performance Analysis on {model_to_compare}")
            performance_summary = run_benchmark_and_analysis(
                onnx_model=model_to_path(model_to_compare),
                scenario=scenario,
            )
            compare_model_analysis.benchmark_results = [performance_summary]
            summary_comparison_model = compare_model_analysis.summary(
                by_types=by_types,
                by_layers=by_layers,
            )
            print(f"Comparing {model_path} with {model_to_compare}")
            print("Note: comparison analysis displays differences b/w models")
            comparison = summary - summary_comparison_model
            comparison.pretty_print()

    if save:
        _LOGGER.info(f"Writing results to {save}")
        analysis.yaml(file_path=save)


def run_benchmark_and_analysis(
    onnx_model: str,
    scenario: BenchmarkScenario,
    sparsity: Optional[float] = None,
    quantization: bool = False,
) -> BenchmarkResult:
    """
    A utility method to run benchmark and performance analysis on an ONNX model
    based off of specified setup and sparsification info
    :param onnx_model: Local filepath onnx model
    :param scenario: `BenchmarkScenario` object with specification for running
        benchmark on an onnx model
    :param sparsity: Globally imposed sparsity level, should be within (0, 1.0]
    :param quantization: Flag to turn quantization on/off, default is `False`
    :return: A `BenchmarkResult` object encapsulating results from running
        specified benchmark and performance analysis
    """

    benchmark_results = benchmark_model(
        model_path=onnx_model,
        batch_size=scenario.batch_size,
        num_cores=scenario.num_cores,
        scenario=scenario.scenario,
        time=scenario.duration,
        warmup_time=scenario.warmup_duration,
        num_streams=scenario.num_streams,
        quiet=True,
    )
    input_list = generate_random_inputs(
        onnx_filepath=onnx_model, batch_size=scenario.batch_size
    )
    analysis_results = model_debug_analysis(
        model=onnx_model,
        inp=input_list,
        batch_size=scenario.batch_size,
        num_cores=scenario.num_cores,
        imposed_ks=sparsity,
        num_warmup_iterations=scenario.warmup_duration,
    )

    items_per_second: float = benchmark_results.get("benchmark_result", {}).get(
        "items_per_sec", 0.0
    )
    average_latency: float = (
        1000.0 / items_per_second if items_per_second > 0 else float("inf")
    )

    node_timings = _get_node_timings_from_analysis_results(
        onnx_model_file=onnx_model, analysis_results=analysis_results
    )
    supported_graph_percentage = benchmark_results.get("fraction_of_supported_ops")
    # TODO: Add recipe info
    imposed_sparsification = ImposedSparsificationInfo(
        sparsity=sparsity,
        quantization=quantization,
    )
    results = BenchmarkResult(
        setup=scenario,
        imposed_sparsification=imposed_sparsification,
        items_per_second=items_per_second,
        average_latency=average_latency,
        node_timings=node_timings,
        supported_graph_percentage=supported_graph_percentage or 0.0,
    )
    return results


def _get_node_timings_from_analysis_results(
    onnx_model_file: str, analysis_results: Dict[str, Any]
) -> Optional[List[NodeInferenceResult]]:
    if not (analysis_results and analysis_results.get("layer_info")):
        return None

    model: ModelProto = onnx.load(onnx_model_file)
    canonical_name_to_node_name = {
        output: node.name for node in model.graph.node for output in node.output
    }

    # dereference node names
    for node in model.graph.node:
        name = node.name
        if node.op_type == "BatchNormalization":
            potential_parents = node.input
            for parent in potential_parents:
                if "Conv" in canonical_name_to_node_name.get(parent, ""):
                    name = canonical_name_to_node_name[parent]
        for output in node.output:
            canonical_name_to_node_name[output] = name

    layer_info = analysis_results["layer_info"]
    node_timings: List[NodeInferenceResult] = []

    for layer in layer_info:
        canonical_name = layer.get("canonical_name")
        if canonical_name in canonical_name_to_node_name:
            name = canonical_name_to_node_name[canonical_name]

            layer_copy = copy.copy(layer)
            avg_run_time = layer_copy.pop("average_run_time_in_ms", None)

            node_timing = NodeInferenceResult(
                name=name,
                avg_run_time=avg_run_time,
                extras=layer_copy,
            )
            node_timings.append(node_timing)

    return node_timings


if __name__ == "__main__":
    main()
