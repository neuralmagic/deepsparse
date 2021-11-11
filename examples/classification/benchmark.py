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
An example script to benchmark a classification model

Supports ONNX files or SparseZoo stubs for models
Supports benchmarking on random or provided data
Supports overriding original input shape
Supports both onnxruntime and DeepSparse runtimes

#########
Command help:
usage: benchmark.py [-h] [-e {deepsparse,onnxruntime}] [--data-path DATA_PATH]
                    [--image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]]
                    [-b BATCH_SIZE] [-c NUM_CORES]
                    [-i NUM_ITERATIONS] [-w NUM_WARMUP_ITERATIONS]
                    model_filepath

Benchmark classification model

positional arguments:
  model_filepath        The full filepath of the ONNX model file or SparseZoo
                        stub to the model for DeepSparse or ONNX runtime
                        benchmarks.

optional arguments:
  -h, --help            show this help message and exit
  -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                        Inference engine backend to run benchmark on. Choices
                        are 'deepsparse', and 'onnxruntime'. Default is
                        'deepsparse'
  --data-path DATA_PATH
                        path to SparseZoo stub for data sample to benchmark
                        with, or to directory of .npz files with named sample
                        inputs to the given model, inputs should be for a
                        single sample with no batch dimension included.
                        Defaults to benchmark on random image samples with
                        channel size 3.
  --image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Image shape to benchmark with, must be two integers.
                        If None model is used to infer image shape
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to run the benchmark for
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the benchmark on,
                        defaults to None where it uses all physical cores
                        available on the system. For DeepSparse benchmarks,
                        this value is the number of cores per socket
  -i NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        The number of iterations the benchmark will be run for
  -w NUM_WARMUP_ITERATIONS, --num-warmup-iterations NUM_WARMUP_ITERATIONS
                        The number of warmup iterations that will be executed
                        before the actual benchmarking
#####
Example
python benchmark.py \
    "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate"
"""
import argparse
import time
from tempfile import NamedTemporaryFile
from typing import Any, List, NamedTuple, Optional, Tuple, Union

import numpy
import onnx
import onnxruntime
from tqdm import tqdm

from deepsparse import compile_model
from deepsparse.benchmark import BenchmarkResults
from helper import _BatchLoader, _load_random_data, load_data
from sparseml.onnx.utils import (
    get_tensor_dim_shape,
    override_model_batch_size,
    set_tensor_dim_shape,
)
from sparsezoo import Zoo


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

_Timer = NamedTuple(
    "_Timer",
    [
        ("start", float),
        ("end", float),
    ],
)


def benchmark():
    """
    Driver function for benchmarking a torchvision classification model
    according to provided arguments
    """

    config = _parse_args(arguments=None)
    model, new_image_shape = _load_model(
        model_filepath=config.model_filepath,
        batch_size=config.batch_size,
        num_cores=config.num_cores,
        engine=config.engine,
        image_shape=config.image_shape,
    )

    total_iterations = config.num_iterations + config.num_warmup_iterations
    data_loader = _get_data_loader(config, new_image_shape, total_iterations)

    print(
        f"Running for {config.num_warmup_iterations} warmup iterations "
        f"and {config.num_iterations} benchmarking iterations",
        flush=True,
    )

    results = BenchmarkResults()
    progress_bar = tqdm(total=config.num_iterations)

    for iteration, batch in enumerate(data_loader):

        # inference
        _, _timer = _timed_run(args=config, model=model, batch=batch)

        if iteration >= config.num_warmup_iterations:
            results.append_batch(
                time_start=_timer.start,
                time_end=_timer.end,
                batch_size=config.batch_size,
            )
            progress_bar.update(1)

    progress_bar.close()
    print(f"Benchmarking complete. End-to-end results:\n{results}")


def fix_onnx_input_shape(
    model_path: str,
    image_shape: Optional[Tuple[int]],
) -> Tuple[str, Optional[NamedTemporaryFile]]:
    """
    Creates a new ONNX model from the given path that accepts the given input
    shape. If the given model already has the given input shape no modifications are
    made. Uses a tempfile to store the modified model file.

    :param model_path: file path to ONNX model or SparseZoo stub of the model
        to be loaded
    :param image_shape: 2-tuple of the image shape to resize this model to, or None if
        no resizing needed
    :return: filepath to an onnx model reshaped to the given input shape will be the
        original path if the shape is the same.  Additionally returns the
        NamedTemporaryFile for managing the scope of the object for file deletion.
        Additionally returns the image-shape to benchmark the new model with.
    """
    original_model_path = model_path
    if model_path.startswith("zoo:"):
        # load SparseZoo Model from stub
        model = Zoo.load_model_from_stub(model_path)
        model_path = model.onnx_file.downloaded_path()
        print(f"Downloaded {original_model_path} to {model_path}")

    model = onnx.load(model_path)
    model_input = model.graph.input[0]

    original_x = get_tensor_dim_shape(model_input, 2)
    original_y = get_tensor_dim_shape(model_input, 3)
    original_image_shape = (original_x, original_y)

    if image_shape is None or original_image_shape == tuple(image_shape):
        return model_path, None, original_image_shape  # no shape modification needed

    set_tensor_dim_shape(model_input, 2, image_shape[0])
    set_tensor_dim_shape(model_input, 3, image_shape[1])

    tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
    onnx.save(model, tmp_file.name)

    print(
        f"Overwriting original model shape {original_image_shape} to {image_shape}\n"
        f"Original model path: {original_model_path}, new temporary model saved to "
        f"{tmp_file.name}"
    )

    return tmp_file.name, tmp_file, image_shape


def _timed_run(
    args, model: Any, batch: Union[numpy.ndarray]
) -> Tuple[List[Union[numpy.ndarray]]]:
    # run model according to engine type
    if args.engine == ORT_ENGINE:
        outputs_ = [out.name for out in model.get_outputs()]
        inputs_ = {model.get_inputs()[0].name: batch}
        start_time = time.time()
        outputs = model.run(
            outputs_,  # outputs
            inputs_,  # inputs dict
        )
        end_time = time.time()

    else:  # deepsparse
        start_time = time.time()
        outputs = model.run(batch)
        end_time = time.time()
    return outputs, _Timer(start=start_time, end=end_time)


def _get_data_loader(config, new_image_shape, total_iterations):
    # Helper function to get appropriate batch data loader
    if not config.data_path:
        data_loader = _load_random_data(
            batch_size=config.batch_size,
            iterations=total_iterations,
            image_size=new_image_shape,
        )
    else:
        dataset = load_data(config.data_path)
        data_loader = _BatchLoader(dataset, config.batch_size, total_iterations)
    return data_loader


def _load_model(
    model_filepath: str,
    batch_size: int,
    num_cores: Optional[int],
    engine: str,
    image_shape: Tuple[int, int],
):
    # load and return respective classification model according to arguments

    if (
        num_cores is not None
        and engine == ORT_ENGINE
        and onnxruntime.__version__ < "1.7"
    ):
        raise ValueError(
            "overriding default num_cores not supported for onnxruntime < 1.7.0. "
            "If using an older build with OpenMP, try setting the OMP_NUM_THREADS "
            "environment variable"
        )

    # scale static ONNX graph to desired image shape
    if engine in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        model_filepath, _, image_shape = fix_onnx_input_shape(
            model_filepath, image_shape
        )

    # load model
    if engine == DEEPSPARSE_ENGINE:
        print(f"Compiling deepsparse model for {model_filepath}")
        model = compile_model(model_filepath, batch_size, num_cores)
        print(f"Engine info: {model}")

    elif engine == ORT_ENGINE:
        print(f"loading onnxruntime model for {model_filepath}")

        sess_options = onnxruntime.SessionOptions()
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        onnx_model = onnx.load(model_filepath)
        override_model_batch_size(onnx_model, batch_size)
        model = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )

    return model, image_shape


def _parse_args(arguments=None):
    parser = argparse.ArgumentParser(description="Benchmark classification model")
    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full filepath of the ONNX model file or SparseZoo stub to the model "
            "for DeepSparse or ONNX runtime benchmarks. "
        ),
    )
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[
            DEEPSPARSE_ENGINE,
            ORT_ENGINE,
        ],
        help=(
            f"Inference engine backend to run benchmark on. Choices are "
            f"'{DEEPSPARSE_ENGINE}', and '{ORT_ENGINE}'. "
            f"Default is '{DEEPSPARSE_ENGINE}'"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help=(
            "path to SparseZoo stub for data sample to benchmark with, or to directory "
            "of .npz files with named sample inputs to the given model, inputs should "
            "be for a single sample with no batch dimension included. "
            "Defaults to benchmark on random image samples with channel size 3."
        ),
        default=None,
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        default=None,
        nargs="+",
        help="Image shape to benchmark with, must be two integers. "
        "If None model is used to infer image shape",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to run the benchmark for",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the benchmark on, "
            "defaults to None where it uses all physical cores available on the system."
            " For DeepSparse benchmarks, this value is the number of cores per socket"
        ),
    )
    parser.add_argument(
        "-i",
        "--num-iterations",
        help="The number of iterations the benchmark will be run for",
        type=int,
        default=80,
    )
    parser.add_argument(
        "-w",
        "--num-warmup-iterations",
        help=(
            "The number of warmup iterations that will be executed before the actual"
            " benchmarking"
        ),
        type=int,
        default=25,
    )

    args = parser.parse_args(args=arguments)
    return args


if __name__ == "__main__":
    benchmark()
