import argparse

import numpy

import onnxruntime
from deepsparse import compile_model, cpu
from deepsparse.utils import verify_outputs
from deepsparse.utils.onnx import (
    generate_random_inputs,
    get_input_names,
    get_output_names,
)


NUM_PHYSICAL_CORES, AVX_TYPE, _ = cpu.cpu_details()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark an ONNX model, comparing between DeepSparse and ONNXRuntime"
    )

    parser.add_argument(
        "onnx-filepath",
        help="The full filepath of the onnx model file being benchmarked",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-j",
        "--num-cores",
        type=int,
        default=NUM_PHYSICAL_CORES,
        help="The number of physical cores to run the analysis on, "
        "defaults to all physical cores available on the system",
    )

    return parser.parse_args()


def main(args):
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_cores = args.num_cores

    inputs = generate_random_inputs(onnx_filepath, batch_size)

    # Gather ONNXRuntime outputs
    print("Executing model with ONNXRuntime...")
    ort_network = onnxruntime.InferenceSession(onnx_filepath)
    ort_outputs = ort_network.run(
        get_output_names(onnx_filepath),
        {name: value for name, value in zip(get_input_names(onnx_filepath), inputs)},
    )

    # Gather NM outputs
    print("Executing model with DeepSparse...")
    nm_network = compile_model(onnx_filepath, batch_size, num_cores)
    nm_outputs = nm_network(inputs)

    max_diffs = verify_outputs(nm_outputs, ort_outputs)

    print("SUCCESS: NM output matches ONNXRuntime output")


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
