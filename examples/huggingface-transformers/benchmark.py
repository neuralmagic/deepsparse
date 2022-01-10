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
Benchmarking script for BERT ONNX models with the DeepSparse engine.


##########
Command help:
usage: benchmark.py [-h] [--data-path DATA_PATH] [-e {deepsparse,onnxruntime}]
                    [-b BATCH_SIZE] [-c NUM_CORES]
                    [-i NUM_ITERATIONS] [-w NUM_WARMUP_ITERATIONS]
                    [--max-sequence-length MAX_SEQUENCE_LENGTH]
                    model_filepath

Benchmark sparsified transformer models

positional arguments:
  model_filepath        The full filepath of the ONNX model file or SparseZoo
                        stub to the model for deepsparse and onnxruntime
                        benchmarks

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        path to sparsezoo stub for data sample to benchmark
                        with, or to directory of .npz files with named sample
                        inputs to the given model, inputs should be for a
                        single sample with no batch dimension included.
                        Defaults to load data samples from a question-
                        answering bert model in sparsezoo
  -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                        Inference engine backend to run benchmark on. Choices
                        are 'deepsparse', 'onnxruntime'. Default is
                        'deepsparse'
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
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        the sequence length to benchmark with. Default is 128

##########
Example for benchmarking on a pruned BERT model from sparsezoo with deepsparse:
python benchmark.py \
   zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98 \

##########
Example for benchmarking on a local ONNX model with deepsparse:
python benchmark.py \
    /PATH/TO/bert.onnx \
    --batch-size 1 \

##########
Example for benchmarking on a local ONNX model with onnxruntime:
python benchmark.py \
    /PATH/TO/bert.onnx \
    --engine onnxruntime \
    --batch-size 32 \
"""

import argparse
import os
import time
import warnings
from typing import Any, Generator, List, Tuple

import numpy
from tqdm.auto import tqdm

from deepsparse import compile_model
from deepsparse.benchmark import BenchmarkResults
from deepsparse.transformers import overwrite_transformer_onnx_model_inputs
from sparsezoo import Zoo
from sparsezoo.utils import load_numpy, load_numpy_list


try:
    import onnxruntime

    ort_error = None
except Exception as err:
    onnxruntime = None
    ort_error = err


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark sparsified transformer models"
    )

    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full filepath of the ONNX model file or SparseZoo stub to the model "
            "for deepsparse and onnxruntime benchmarks"
        ),
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help=(
            "path to sparsezoo stub for data sample to benchmark with, or to directory "
            "of .npz files with named sample inputs to the given model, inputs should "
            "be for a single sample with no batch dimension included. Defaults to load "
            "data samples from a question-answering bert model in sparsezoo"
        ),
        default=(
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"
        ),
    )

    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[DEEPSPARSE_ENGINE, ORT_ENGINE],
        help=(
            "Inference engine backend to run benchmark on. Choices are 'deepsparse', "
            "'onnxruntime'. Default is 'deepsparse'"
        ),
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
    parser.add_argument(
        "--max-sequence-length",
        help="the sequence length to benchmark with. Default is 128",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    return args


def benchmark(args):
    """
    Method to benchmark inference times for BERT and transformer models
    """
    model, input_names = _load_model(args)
    dataset = _load_data(args, input_names)

    print(
        (
            f"Running for {args.num_warmup_iterations} warmup iterations "
            f"and {args.num_iterations} benchmarking iterations"
        ),
        flush=True,
    )

    results = BenchmarkResults()
    total_iterations = args.num_warmup_iterations + args.num_iterations
    data_loader = _iter_batches(dataset, args.batch_size, total_iterations)
    progress_bar = tqdm(total=args.num_iterations)

    for iteration, batch in enumerate(data_loader):
        iter_start = time.time()

        # inference
        _ = _run_model(args, model, batch, input_names)

        iter_end = time.time()

        if iteration >= args.num_warmup_iterations:
            results.append_batch(
                time_start=iter_start,
                time_end=iter_end,
                batch_size=args.batch_size,
            )
            progress_bar.update(1)

    progress_bar.close()

    print(f"Benchmarking complete. End-to-end results:\n{results}")


def _load_model(args) -> Tuple[Any, List[str]]:
    if args.engine == ORT_ENGINE and ort_error is not None:
        raise ort_error

    # validation
    if (
        args.num_cores is not None
        and args.engine == ORT_ENGINE
        and onnxruntime.__version__ < "1.7"
    ):
        raise ValueError(
            "overriding default num_cores not supported for onnxruntime < 1.7.0. "
            "If using an older build with OpenMP, try setting the OMP_NUM_THREADS "
            "environment variable"
        )

    # load model from sparsezoo if necessary
    if args.model_filepath.startswith("zoo:"):
        zoo_model = Zoo.load_model_from_stub(args.model_filepath)
        downloaded_path = zoo_model.onnx_file.downloaded_path()
        print(f"downloaded sparsezoo model {args.model_filepath} to {downloaded_path}")
        args.model_filepath = downloaded_path

    # scale static ONNX graph to desired image shape
    input_names = []
    if args.engine in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        args.model_filepath, input_names, _ = overwrite_transformer_onnx_model_inputs(
            args.model_filepath,
            batch_size=args.batch_size,
            max_length=args.max_sequence_length,
        )

    # load model
    if args.engine == DEEPSPARSE_ENGINE:
        print(f"Compiling deepsparse model for {args.model_filepath}")
        model = compile_model(args.model_filepath, args.batch_size, args.num_cores)
        print(f"Engine info: {model}")
    elif args.engine == ORT_ENGINE:
        print(f"loading onnxruntime model for {args.model_filepath}")

        sess_options = onnxruntime.SessionOptions()
        if args.num_cores is not None:
            sess_options.intra_op_num_threads = args.num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        model = onnxruntime.InferenceSession(
            args.model_filepath, sess_options=sess_options
        )

    return model, input_names


def _load_data(args, input_names) -> List[List[numpy.ndarray]]:
    if args.data_path.startswith("zoo:"):
        data_dir = Zoo.load_model_from_stub(
            args.data_path
        ).data_inputs.downloaded_path()
    else:
        data_dir = args.data_path
        data_files = os.listdir(data_dir)
        if any(".npz" not in file_name for file_name in data_files):
            raise RuntimeError(
                f"All files in data directory {data_dir} must have a .npz extension "
                f"found {[name for name in data_files if '.npz' not in name]}"
            )

    samples = load_numpy_list(data_dir)

    # unwrap unloaded numpy files
    samples = [
        load_numpy(sample) if isinstance(sample, str) else sample for sample in samples
    ]

    processed_samples = []
    warning_given = False
    for sample in samples:
        if not all(inp_name in sample for inp_name in input_names) or len(
            input_names
        ) != len(sample):
            if not warning_given:
                warnings.warn(
                    "input sample found whose input names do not match the model input "
                    "names, this may cause an exception during benchmarking"
                )
                warning_given = True
            sample = list(sample.values())
        else:
            sample = [sample[inp_name] for inp_name in input_names]

        for idx, array in enumerate(sample):
            processed_array = numpy.zeros(
                [args.max_sequence_length, *array.shape[1:]],
                dtype=array.dtype,
            )
            if array.shape[0] < args.max_sequence_length:
                processed_array[: array.shape[0], ...] = array
            else:
                processed_array[:, ...] = array[: args.max_sequence_length, ...]
            sample[idx] = processed_array
        processed_samples.append(sample)
    return processed_samples


def _iter_batches(
    dataset: List[List[numpy.ndarray]],
    batch_size: int,
    iterations: int,
) -> Generator[List[numpy.ndarray], None, None]:
    iteration = 0
    batch_buffer = []
    batch_template = [
        numpy.ascontiguousarray(
            numpy.zeros((batch_size, *array.shape), dtype=array.dtype)
        )
        for array in dataset[0]
    ]
    while iteration < iterations:
        for sample in dataset:
            batch_buffer.append(sample)

            if len(batch_buffer) == batch_size:
                yield [
                    numpy.stack([sample[idx] for sample in batch_buffer], out=template)
                    for idx, template in enumerate(batch_template)
                ]

                batch_buffer = []
                iteration += 1

                if iteration >= iterations:
                    break


def _run_model(
    args, model: Any, batch: List[numpy.ndarray], input_names: List[str]
) -> List[numpy.ndarray]:
    outputs = None
    if args.engine == ORT_ENGINE:
        outputs = model.run(
            None,
            dict(zip(input_names, batch)),  # inputs dict
        )
    elif args.engine == DEEPSPARSE_ENGINE:  # deepsparse
        outputs = model.run(batch)
    return outputs


def main():
    _config = parse_args()
    benchmark(args=_config)


if __name__ == "__main__":
    main()
