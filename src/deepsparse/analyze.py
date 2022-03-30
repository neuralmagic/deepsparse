

import argparse
import sys
import pprint

from deepsparse import analyze_model
from deepsparse.utils import (
    generate_random_inputs,
    model_to_path,
    override_onnx_input_shapes,
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
        default=0,
    )
    parser.add_argument(
        "-bi",
        "--num_iterations",
        help="The number of times the benchmark will be run",
        type=int,
        default=1,
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
        "-v",
        "--verbose",
        help="Print all of the benchmark info",
        action="store_true",
        default=False,
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
        "-i",
        "--input_shapes",
        help="Override the shapes of the inputs, "
        'i.e. -shapes "[1,2,3],[4,5,6],[7,8,9]" results in '
        "input0=[1,2,3] input1=[4,5,6] input2=[7,8,9]",
        type=str,
        default="",
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
        "Name                        | OutDims                    | KerDims                    |"
        + " Strides      | ActSpars | Time(ms) |  Util(%) | TFLOPS   | Canonical Name\n"
    )
    for li in result["layer_info"]:
        table_str += layer_info_to_string(
            li,
            "{:28}| {:26} | {:26} | {:12} | {: >#08.4f} | {: >#08.4f} | {: >#08.4f} | {: >#08.4f} | {:12}\n",
        )
        for sub_li in li["sub_layer_info"]:
            table_str += layer_info_to_string(
                sub_li,
                "  {:26}| {:26} | {:26} | {:12} | {: >#08.4f} | {: >#08.4f} | {: >#08.4f} | {: >#08.4f} | {:12}\n",
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

    random_input = args.random_input
    num_warmup_iterations = args.num_warmup_iterations
    num_iterations = args.num_iterations
    num_cores = args.num_cores
    batch_size = args.batch_size
    imposed_activation_sparsity = args.activation_sparsity
    imposed_kernel_sparsity = args.kernel_sparsity
    imposed_kernel_sparsity_file = args.kernel_sparsity_file
    optimization_level = args.optimization
    input_shapes = parse_input_shapes(args.input_shapes)
    verbose = args.verbose

    # Imposed KS can take either a float or a file, so overwrite with file if we have it
    if imposed_kernel_sparsity_file:
        imposed_kernel_sparsity = imposed_kernel_sparsity_file

    orig_model_path = args.model_path
    model_path = model_to_path(args.model_path)

    print("onnx_filename: " + str(model_path))

    if input_shapes:
        with override_onnx_input_shapes(model_path, input_shapes) as tmp_path:
            input_list = generate_random_inputs(tmp_path, batch_size)
    else:
        input_list = generate_random_inputs(model_path, batch_size)

    result = analyze_model(
        model_path,
        input_list,
        batch_size=batch_size,
        num_cores=num_cores,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        optimization_level=optimization_level,
        imposed_as=imposed_activation_sparsity,
        imposed_ks=imposed_kernel_sparsity,
        input_shapes=input_shapes,
    )

    if verbose:
        pprint.pprint(result)

    print(construct_layer_table(result))
    print(construct_layer_statistics(result))

if __name__ == "__main__":
    main()