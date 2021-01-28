"""
Example script for generating random input from an ONNX model and running
the model both through the DeepSparse Engine and ONNXRuntime, comparing
outputs to confirm they are the same.

In this method, we can assume that ONNXRuntime will give the
"correct" output as it is the industry-standard solution.

##########
Command help:
usage: check_correctness.py [-h] [-s BATCH_SIZE] [-j NUM_CORES] onnx_filepath

Run an ONNX model, comparing outputs between the DeepSparse Engine and ONNXRuntime

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file being run

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -j NUM_CORES, --num_cores NUM_CORES
                        The number of physical cores to run the analysis on, defaults to all physical cores available on the system

##########
Example command for checking a downloaded resnet50 model for batch size 8 and 4 cores:
python examples/benchmark/check_correctness.py \
    ~/Downloads/resnet50.onnx \
    --batch_size 8 \
    --num_cores 4
"""

import argparse

import onnxruntime

from deepsparse import compile_model, cpu
from deepsparse.utils import (
    generate_random_inputs,
    get_input_names,
    get_output_names,
    override_onnx_batch_size,
    verify_outputs,
)


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an ONNX model, comparing outputs between the DeepSparse Engine and ONNXRuntime"
    )

    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file being run",
    )

    parser.add_argument(
        "-s",
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-j",
        "--num_cores",
        type=int,
        default=CORES_PER_SOCKET,
        help="The number of physical cores to run the analysis on, "
        "defaults to all physical cores available on the system",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_cores = args.num_cores

    inputs = generate_random_inputs(onnx_filepath, batch_size)
    input_names = get_input_names(onnx_filepath)
    output_names = get_output_names(onnx_filepath)
    inputs_dict = {name: value for name, value in zip(input_names, inputs)}

    # ONNXRuntime inference
    print("Executing model with ONNXRuntime...")
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = num_cores
    with override_onnx_batch_size(onnx_filepath, batch_size) as override_onnx_filepath:
        ort_network = onnxruntime.InferenceSession(override_onnx_filepath, sess_options)

        ort_outputs = ort_network.run(output_names, inputs_dict)

    # DeepSparse Engine inference
    print("Executing model with DeepSparse Engine...")
    dse_network = compile_model(onnx_filepath, batch_size, num_cores)
    dse_outputs = dse_network(inputs)

    verify_outputs(dse_outputs, ort_outputs)

    print("DeepSparse Engine output matches ONNXRuntime output")


if __name__ == "__main__":
    main()
