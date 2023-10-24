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
Analysis script for ONNX models with the DeepSparse engine.

##########
Command help:
usage: deepsparse.debug_analysis [-h] [-wi NUM_WARMUP_ITERATIONS]
                                 [-bi NUM_ITERATIONS] [-ncores NUM_CORES]
                                 [-b BATCH_SIZE] [-i INPUT_SHAPES]
                                 [-ks KERNEL_SPARSITY]
                                 [-ksf KERNEL_SPARSITY_FILE]
                                 [--optimization OPTIMIZATION]
                                 [-seq_len SEQUENCE_LENGTH]
                                 [-input_ids_len INPUT_IDS_LENGTH]
                                 [--no-internal-kv-cache]
                                 [--kv-cache-prev-num-tokens KV_CACHE_PREV_NUM_TOKENS]
                                 [--kv-cache-num-frozen-tokens KV_CACHE_NUM_FROZEN_TOKENS]
                                 [-q] [-x EXPORT_PATH]
                                 [--disable-kv-cache-overrides]
                                 model_path

Analyze ONNX models in the DeepSparse Engine

positional arguments:
  model_path            Path to an ONNX model file or SparseZoo model stub

optional arguments:
  -h, --help            show this help message and exit
  -wi NUM_WARMUP_ITERATIONS, --num_warmup_iterations NUM_WARMUP_ITERATIONS
                        The number of warmup runs that will be executed before
                        the actual benchmarking
  -bi NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
                        The number of times the benchmark will be run
  -ncores NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The number of inputs that will run through the model
                        at a time
  -i INPUT_SHAPES, -shapes INPUT_SHAPES, --input_shapes INPUT_SHAPES
                        Override the shapes of the inputs, i.e. -shapes
                        "[1,2,3],[4,5,6],[7,8,9]" results in input0=[1,2,3]
                        input1=[4,5,6] input2=[7,8,9]
  -ks KERNEL_SPARSITY, --kernel_sparsity KERNEL_SPARSITY
                        Impose kernel sparsity for all convolutions. [0.0-1.0]
  -ksf KERNEL_SPARSITY_FILE, --kernel_sparsity_file KERNEL_SPARSITY_FILE
                        Filepath to per-layer kernel sparsities JSON
  --optimization OPTIMIZATION
                        To enable or disable optimizations (Tensor Columns)
  -seq_len SEQUENCE_LENGTH, --sequence_length SEQUENCE_LENGTH
                        The sequence length to run the KV cache supported
                        model benchmarks for. Must be seq_len >= 1, default is
                        None
  -input_ids_len INPUT_IDS_LENGTH, --input_ids_length INPUT_IDS_LENGTH
                        The input ids length to run the KV cache supported
                        model benchmarks for. Must be 1 <= input_ids_len <=
                        seq_len, default is 1
  --no-internal-kv-cache, --no_internal_kv_cache
                        If not present, and model has KV cache, KV Cache state
                        will be managed within the compiled deepsparse model.
                        This is preferred when applicable for best
                        performance. Set flag to disable
  --kv-cache-prev-num-tokens KV_CACHE_PREV_NUM_TOKENS
                        Internal KVCache: The amount of previous tokens that
                        will be read from the external KV cache on the first
                        inference
  --kv-cache-num-frozen-tokens KV_CACHE_NUM_FROZEN_TOKENS
                        Internal KVCache: The amount of first tokens that we
                        want to keep permanently in the KV cache
  -q, --quiet           Lower logging verbosity
  -x EXPORT_PATH, --export_path EXPORT_PATH
                        Store results into a JSON or CSV file
  --disable-kv-cache-overrides, --disable_kv_cache_overrides
                        If set, it will not alter the model
                        with kv cache overrides
"""  # noqa E501

import argparse
import json
import os

from deepsparse import KVCacheParams, model_debug_analysis
from deepsparse.utils import (
    default_cached_outputs,
    generate_random_inputs,
    has_model_kv_cache,
    infer_sequence_length,
    model_to_path,
    override_onnx_input_shapes,
    overwrite_onnx_model_inputs_for_kv_cache_models,
    parse_input_shapes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze ONNX models in the DeepSparse Engine"
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to an ONNX model file or SparseZoo model stub",
    )
    parser.add_argument(
        "-wi",
        "--num_warmup_iterations",
        help="The number of warmup runs that will be executed before the \
        actual benchmarking",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-bi",
        "--num_iterations",
        help="The number of times the benchmark will be run",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-ncores",
        "--num_cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the analysis on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="The number of inputs that will run through the model at a time",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-i",
        "-shapes",
        "--input_shapes",
        help="Override the shapes of the inputs, "
        'i.e. -shapes "[1,2,3],[4,5,6],[7,8,9]" results in '
        "input0=[1,2,3] input1=[4,5,6] input2=[7,8,9]",
        type=str,
        default="",
    )
    parser.add_argument(
        "-ks",
        "--kernel_sparsity",
        help="Impose kernel sparsity for all convolutions. [0.0-1.0]",
        type=float,
    )
    parser.add_argument(
        "-ksf",
        "--kernel_sparsity_file",
        help="Filepath to per-layer kernel sparsities JSON",
        type=str,
    )
    parser.add_argument(
        "--optimization",
        help="To enable or disable optimizations (Tensor Columns)",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-seq_len",
        "--sequence_length",
        type=int,
        default=None,
        help="The sequence length to run the KV cache supported model "
        "benchmarks for. Must be seq_len >= 1, default is None",
    )
    parser.add_argument(
        "-input_ids_len",
        "--input_ids_length",
        type=int,
        default=1,
        help="The input ids length to run the KV cache supported model "
        "benchmarks for. Must be 1 <= input_ids_len <= seq_len, default is 1",
    )
    parser.add_argument(
        "--no-internal-kv-cache",
        "--no_internal_kv_cache",
        help=(
            "If not present, and model has KV cache, "
            "KV Cache state will be managed within the compiled deepsparse "
            "model. This is preferred when applicable for best performance. Set "
            "flag to disable"
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--kv-cache-prev-num-tokens",
        help="Internal KVCache: The amount of previous tokens that will be read"
        " from the external KV cache on the first inference",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--kv-cache-num-frozen-tokens",
        help="Internal KVCache: The amount of first tokens that we want to keep"
        " permanently in the KV cache",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Lower logging verbosity",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-x",
        "--export_path",
        help="Store results into a JSON or CSV file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--disable-kv-cache-overrides",
        "--disable_kv_cache_overrides",
        help=(
            "If set, deepsparse.benchmark will not alter the model "
            "with kv cache overrides"
        ),
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def layer_info_to_string(li, format_str):
    if li["name"] == "sub_pyramid":
        return format_str.format(li["name"], "[]", "[]", "[]", 0, 0, 0, 0, "")
    else:
        return format_str.format(
            li["name"],
            "{}".format(list(li["output_dims"].values())),
            "{}".format(list(li["kernel_dims"].values())),
            "{}".format(list(li["strides"].values())),
            li["activation_sparsity"],
            li["average_run_time_in_ms"],
            li["average_utilization"] * 100.0,
            li["average_teraflops_per_second"],
            li["canonical_name"],
        )


def construct_layer_table(result):
    table_str = (
        "Name                        | OutDims                    | "
        "KerDims                    | Strides      | ActSpars | "
        "Time(ms) |  Util(%) | TFLOPS   | Canonical Name\n"
    )
    info_format_base = (
        "{:26} | {:26} | {:12} | {: >#08.4f} | "
        "{: >#08.4f} | {: >#08.4f} | {: >#08.4f} | {:12}"
    )
    for li in result["layer_info"]:
        table_str += layer_info_to_string(li, "{:28}| " + info_format_base + "\n")
        for sub_li in li["sub_layer_info"]:
            table_str += layer_info_to_string(
                sub_li, "  {:26}| " + info_format_base + "\n"
            )

    table_str += "Total Time(MS): {:05f}\n".format(result["average_total_time"])
    table_str += "Items per second: {:05f}\n".format(result["items_per_second"])
    table_str += "Batch Size: {}\n".format(result["batch_size"])
    table_str += "Number of threads: {}\n".format(result["num_threads"])

    return table_str


def process_line_item(total_layer_time, detailed_layer_time, li, strip_name):
    if "average_run_time_in_ms" not in li:
        # nothing to process
        return

    layer_type = li["name"]
    if strip_name:
        # peel off unique number
        layer_type = layer_type.rsplit("_", 1)[0]
        # peel off ks percentage
        layer_type = layer_type.rsplit("-", 1)[0]

    avg_layer_time = li["average_run_time_in_ms"]

    if layer_type in total_layer_time:
        total_layer_time[layer_type] += avg_layer_time
    else:
        total_layer_time[layer_type] = avg_layer_time

    # Record detailed layer types as well
    if "kernel_dims" in li:
        kerdims = list(li["kernel_dims"].values())
        if kerdims:
            detailed_layer_type = f"{layer_type}|kernel={kerdims}"
            if detailed_layer_type in detailed_layer_time:
                detailed_layer_time[detailed_layer_type] += avg_layer_time
            else:
                detailed_layer_time[detailed_layer_type] = avg_layer_time


def construct_layer_statistics(result):
    # Percentage Statistics
    total_layer_time = {}
    detailed_layer_time = {}
    for li in result["layer_info"]:
        if len(li["sub_layer_info"]) == 0:
            process_line_item(total_layer_time, detailed_layer_time, li, True)
        else:
            for sli in li["sub_layer_info"]:
                process_line_item(total_layer_time, detailed_layer_time, sli, False)

    summed_total_time = 0.0
    for k, v in total_layer_time.items():
        summed_total_time += v

    perc_str = "== Layer Breakdown ==\n"
    perc_str += "Name                           | Summed Time | Percent Taken\n"
    for name, val in total_layer_time.items():
        # Print summary for this type of layer
        perc_str += "{:30} | {:8.3f}    | {:4.2f}%\n".format(
            name, val, (val / summed_total_time) * 100.0
        )

        # Do the same for any sub-types recorded (there can be none)
        sublayers = [
            (key.split("|", 1)[1], value)
            for key, value in detailed_layer_time.items()
            if name == key.split("|", 1)[0]
        ]
        for subname, subval in sublayers:
            perc_str += "  {:28} | {:8.3f}    | {:4.2f}%\n".format(
                subname, subval, (subval / summed_total_time) * 100.0
            )

    batch_size = int(result["batch_size"])
    perc_str += "== Summed Total Time: {:.4f} ms\n".format(summed_total_time)
    perc_str += "== Items per second: {:.4f}\n".format(
        (1000.0 / summed_total_time) * batch_size
    )

    return perc_str


def main():
    args = parse_args()

    input_shapes = parse_input_shapes(args.input_shapes)

    if args.optimization:
        os.environ["WAND_ENABLE_SP_BENCH"] = "1"

    # Imposed KS can take either a float or a file, so overwrite with file if we have it
    imposed_kernel_sparsity = args.kernel_sparsity
    if args.kernel_sparsity_file:
        imposed_kernel_sparsity = args.kernel_sparsity_file

    orig_model_path = args.model_path
    model_path = model_to_path(args.model_path)

    print(f"Analyzing model: {orig_model_path}")

    batch_size = args.batch_size

    if not args.disable_kv_cache_overrides and has_model_kv_cache(model_path):
        if not args.sequence_length:
            args.sequence_length = infer_sequence_length(model_path)
        if args.input_ids_length > args.sequence_length:
            raise ValueError(
                f"input_ids_length: {args.input_ids_length} "
                f"must be less than sequence_length: {args.sequence_length}"
            )

        print(
            "Found model with KV cache support. "
            "Benchmarking the autoregressive model with "
            f"input_ids_length: {args.input_ids_length} and "
            f"sequence length: {args.sequence_length}."
        )

        model_path, _, _ = overwrite_onnx_model_inputs_for_kv_cache_models(
            onnx_file_path=model_path,
            input_ids_length=args.input_ids_length,
            sequence_length=args.sequence_length,
            batch_size=batch_size,
        )

    if input_shapes:
        with override_onnx_input_shapes(model_path, input_shapes) as tmp_path:
            input_list = generate_random_inputs(tmp_path, batch_size)
    else:
        input_list = generate_random_inputs(model_path, batch_size)

    kv_cache_params = None
    if not args.no_internal_kv_cache:
        kv_cache_params = KVCacheParams(
            default_cached_outputs(model_path),
            args.kv_cache_prev_num_tokens,
            args.kv_cache_num_frozen_tokens,
        )

        print(
            f"Enable KVCache: prev_num_tokens = {kv_cache_params.prev_num_tokens}, "
            f"num_frozen_tokens = {kv_cache_params.num_frozen_tokens}"
        )

    result = model_debug_analysis(
        model_path,
        input_list,
        batch_size=args.batch_size,
        num_cores=args.num_cores,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        optimization_level=int(args.optimization),
        imposed_ks=imposed_kernel_sparsity,
        input_shapes=input_shapes,
        kv_cache_params=kv_cache_params,
    )

    if not args.quiet:
        print(construct_layer_table(result))
    print(construct_layer_statistics(result))

    if args.export_path:
        if ".csv" in args.export_path:
            top_level_items_skip = ["iteration_times", "layer_info"]
            top_level_items_dict = {
                k: v for k, v in result.items() if k not in top_level_items_skip
            }

            def construct_csv_layer_info(li):
                def flatten(parent_k, sub_d):
                    return {f"{parent_k}_{k}": v for k, v in sub_d.items()}

                csv_li = {}
                for k, v in li.items():
                    if k not in ["sub_layer_info"]:
                        csv_li.update({k: v} if type(v) is not dict else flatten(k, v))
                return csv_li

            csv_layer_infos = [
                {
                    **top_level_items_dict,
                    **construct_csv_layer_info(li),
                }
                for li in result["layer_info"]
            ]
            col_keys = {k for li in csv_layer_infos for k in li.keys()}

            # Export results
            import csv

            print("Saving analysis results to CSV file at {}".format(args.export_path))
            with open(args.export_path, "w") as out:
                writer = csv.DictWriter(out, fieldnames=col_keys, extrasaction="ignore")
                writer.writeheader()
                for data in csv_layer_infos:
                    writer.writerow(data)
        else:
            # Export results
            print("Saving analysis results to JSON file at {}".format(args.export_path))
            with open(args.export_path, "w") as out:
                json.dump(result, out, indent=2)


if __name__ == "__main__":
    main()
