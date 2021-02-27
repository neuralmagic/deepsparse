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
Example script for generating random input from an ONNX model and running
the model both through the DeepSparse Engine and ONNXRuntime, comparing
outputs to confirm they are the same.

In this method, we can assume that ONNXRuntime will give the
"correct" output as it is the industry-standard solution.

##########
Command help:
usage: check_correctness.py [-h] [-s BATCH_SIZE] onnx_filepath

Run an ONNX model, comparing outputs between the DeepSparse Engine and ONNXRuntime

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file being run

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for

##########
Example command for checking a downloaded resnet50 model
for batch size 8:
python examples/benchmark/check_correctness.py \
    ~/Downloads/resnet50.onnx \
    --batch_size 8
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
        description=(
            "Run an ONNX model, comparing outputs between the DeepSparse Engine and"
            " ONNXRuntime"
        )
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

    return parser.parse_args()


def main():
    args = parse_args()
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size

    inputs = generate_random_inputs(onnx_filepath, batch_size)
    input_names = get_input_names(onnx_filepath)
    output_names = get_output_names(onnx_filepath)
    inputs_dict = {name: value for name, value in zip(input_names, inputs)}

    # ONNXRuntime inference
    print("Executing model with ONNXRuntime...")
    sess_options = onnxruntime.SessionOptions()
    with override_onnx_batch_size(onnx_filepath, batch_size) as override_onnx_filepath:
        ort_network = onnxruntime.InferenceSession(override_onnx_filepath, sess_options)

        ort_outputs = ort_network.run(output_names, inputs_dict)

    # DeepSparse Engine inference
    print("Executing model with DeepSparse Engine...")
    dse_network = compile_model(onnx_filepath, batch_size=batch_size)
    dse_outputs = dse_network(inputs)

    verify_outputs(dse_outputs, ort_outputs)

    print("DeepSparse Engine output matches ONNXRuntime output")


if __name__ == "__main__":
    main()
