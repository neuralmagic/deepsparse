import argparse

import numpy

import onnxruntime
from deepsparse import compile_model, cpu
from deepsparse.utils import (
    verify_outputs,
    generate_random_inputs,
    get_input_names,
    get_output_names,
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


def main(args):
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_cores = args.num_cores

    inputs = generate_random_inputs(onnx_filepath, batch_size)

    # Gather ONNXRuntime outputs
    print("Executing model with ONNXRuntime...")
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = num_cores
    ort_network = onnxruntime.InferenceSession(onnx_filepath, sess_options)
    ort_outputs = ort_network.run(
        get_output_names(onnx_filepath),
        {name: value for name, value in zip(get_input_names(onnx_filepath), inputs)},
    )

    # Gather DeepSparse Engine outputs
    print("Executing model with DeepSparse Engine...")
    dse_network = compile_model(onnx_filepath, batch_size, num_cores)
    dse_outputs = dse_network(inputs)

    print(ort_outputs)

    print(dse_outputs)

    verify_outputs(dse_outputs, ort_outputs)

    print("DeepSparse Engine output matches ONNXRuntime output")
    print("SUCCESS")


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
