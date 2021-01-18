import argparse
import time

import numpy

import onnxruntime
from deepsparse import BenchmarkResults, compile_model, cpu
from deepsparse.utils import verify_outputs
from deepsparse.utils.onnx import (
    generate_random_inputs,
    get_input_names,
    get_output_names,
)


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark an ONNX model, comparing between DeepSparse and ONNXRuntime"
    )

    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file being benchmarked",
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
    parser.add_argument(
        "-b",
        "--num_iterations",
        help="The number of times the benchmark will be run",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-w",
        "--num_warmup_iterations",
        help="The number of warmup runs that will be executed before the actual benchmarking",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main(args):
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_cores = args.num_cores
    num_iterations = args.num_iterations
    num_warmup_iterations = args.num_warmup_iterations

    inputs = generate_random_inputs(onnx_filepath, batch_size)

    # Benchmark ONNXRuntime
    print("Benchmarking model with ONNXRuntime...")
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = num_cores
    ort_network = onnxruntime.InferenceSession(onnx_filepath, sess_options)
    input_names = get_input_names(onnx_filepath)
    output_names = get_output_names(onnx_filepath)
    inputs_dict = {name: value for name, value in zip(input_names, inputs)}
    ort_outputs = []
    ort_results = BenchmarkResults()
    for i in range(num_warmup_iterations):
        ort_network.run(output_names, inputs_dict)
    for i in range(num_iterations):
        start = time.time()
        output = ort_network.run(output_names, inputs_dict)
        end = time.time()
        ort_results.append_batch(
            time_start=start, time_end=end, batch_size=batch_size, outputs=output
        )

    # Benchmark DeepSparse Engine
    print("Benchmarking model with DeepSparse Engine...")
    dse_network = compile_model(onnx_filepath, batch_size, num_cores)
    dse_results = dse_network.benchmark(
        inputs, num_iterations, num_warmup_iterations, include_outputs=True
    )

    for dse_output, ort_output in zip(dse_results.outputs, ort_results.outputs):
        verify_outputs(dse_output, ort_output)

    print("ONNXRuntime results:")
    print(ort_results)
    print()
    print("DeepSparse Engine results:")
    print(dse_results)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
