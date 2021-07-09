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
usage: benchmark.py [-h] [-e {deepsparse,onnxruntime}] [-b BATCH_SIZE]
                    [-c NUM_CORES] [-s NUM_SOCKETS] [-i NUM_ITERATIONS]
                    [-w NUM_WARMUP_ITERATIONS]
                    model_filepath

Benchmark sparsified transformer models

positional arguments:
  model_filepath        The full filepath of the ONNX model file or SparseZoo
                        stub to the model for deepsparse and onnxruntime
                        benchmarks

optional arguments:
  -h, --help            show this help message and exit
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
  -s NUM_SOCKETS, --num-sockets NUM_SOCKETS
                        For DeepSparse benchmarks only. The number of physical
                        cores to run the benchmark on. Defaults to None where
                        is uses all sockets available on the system
  -i NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        The number of iterations the benchmark will be run for
  -w NUM_WARMUP_ITERATIONS, --num-warmup-iterations NUM_WARMUP_ITERATIONS
                        The number of warmup iterations that will be executed
                        before the actual benchmarking

##########
Example for benchmarking on a pruned BERT model from sparsezoo with deepsparse:
python benchmark.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate \
    --attention-mask-input 1

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
import time
from typing import Any, Generator, Iterable, List

import numpy
import onnx
import onnxruntime
from tqdm.auto import tqdm

from deepsparse import compile_model
from deepsparse.benchmark import BenchmarkResults
from deepsparse.utils import ONNX_TENSOR_TYPE_MAP
from sparseml.onnx.utils import override_model_batch_size


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


def load_random_data():
    """
    Load random tensors for benchmarking bert
    TODO
    """
    pass


def modify_bert_onnx_input_shape(model_filepath):
    """
    Method to scale onnx static graph to desired shape
    TODO
    """
    pass


def _iter_batches(
    dataset: List[Any],
    batch_size: int,
    iterations: int,
) -> Iterable[Any]:
    iteration = 0
    batch = []
    batch_template = numpy.ascontiguousarray(
        numpy.zeros((batch_size, *dataset[0].shape), dtype=dataset[0].dtype)
    )
    while iteration < iterations:
        for item in dataset:
            batch.append(item)

            if len(batch) == batch_size:
                yield numpy.stack(batch, out=batch_template)

                batch = []
                iteration += 1

                if iteration >= iterations:
                    break


def parse_attention_input_name(val):
    if val is None:
        return val
    try:
        val = int(val)
        if val < 0:
            raise ValueError(
                f"attention input idx must be greater than 0. given: {val}"
            )
    except Exception:
        return str(val)


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
        "-s",
        "--num-sockets",
        type=int,
        default=None,
        help=(
            "For DeepSparse benchmarks only. The number of physical cores to run the "
            "benchmark on. Defaults to None where is uses all sockets available on the "
            "system"
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
        "--sequence-length",
        help="the sequence length to benchmark with. Defualt is 128",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--vocab-size",
        help="vocabulary size to create sample model inputs from. Default is 30k",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--attention-mask-input",
        help=(
            "string name or integer index of the attention mask input of the model. "
            "e.g. 'attention_mask' or 1. If provided, the attention mask input will "
            "be set to all ones"
        ),
        type=parse_attention_input_name,
        default=None,
    )

    args = parser.parse_args()

    return args


def benchmark(args):
    """
    Method to benchmark inference times for BERT and transformer models
    """
    model = _load_model(args)
    print("Loading dataset")
    dataset, _ = load_random_data()
    total_iterations = args.num_iterations + args.num_warmup_iterations
    data_loader = _iter_batches(dataset, args.batch_size, total_iterations)

    print(
        (
            f"Running for {args.num_warmup_iterations} warmup iterations "
            f"and {args.num_iterations} benchmarking iterations"
        ),
        flush=True,
    )

    results = BenchmarkResults()
    progress_bar = tqdm(total=args.num_iterations)

    for iteration, batch in enumerate(data_loader):
        iter_start = time.time()

        # inference
        _ = _run_model(args, model, batch)

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
    print(f"End-to-end per image time: {results.ms_per_batch / args.batch_size}ms")


def _load_model(args) -> Any:
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
    if args.num_sockets is not None and args.engine != DEEPSPARSE_ENGINE:
        raise ValueError(f"Overriding num_sockets is not supported for {args.engine}")

    # scale static ONNX graph to desired image shape
    if args.engine in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        args.model_filepath, _ = modify_bert_onnx_input_shape(args.model_filepath)

    # load model
    if args.engine == DEEPSPARSE_ENGINE:
        print(f"Compiling deepsparse model for {args.model_filepath}")
        model = compile_model(
            args.model_filepath, args.batch_size, args.num_cores, args.num_sockets
        )
    elif args.engine == ORT_ENGINE:
        print(f"loading onnxruntime model for {args.model_filepath}")

        sess_options = onnxruntime.SessionOptions()
        if args.num_cores is not None:
            sess_options.intra_op_num_threads = args.num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        onnx_model = onnx.load(args.model_filepath)
        override_model_batch_size(onnx_model, args.batch_size)
        model = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )
    return model


def _iter_random_benchmarking_batches(
    args,
    model_path: str,
    input_names: List[str],
) -> Generator[List[numpy.ndarray], None, None]:
    # extract model input shapes, types, and the attention mask
    model_tmp = onnx.load(model_path)
    is_attention_mask = [False] * len(input_names)
    input_shapes = []
    input_dtypes = []
    for idx, inp in enumerate(model_tmp.graph.input):
        if inp.name not in input_names:
            continue
        if args.attention_mask_input in [idx, inp.name]:
            is_attention_mask[idx] = True
        input_shapes.append([dim.dim_value for dim in inp.type.tensor_type.shape.dim])
        input_dtypes.append(ONNX_TENSOR_TYPE_MAP[inp.type.tensor_type.elem_type])

    # free loaded model from memory
    del model_tmp

    def _generate_sample_input(input_idx):
        if is_attention_mask[input_idx]:
            sample_input = numpy.ones(
                input_shapes[input_idx], dtype=input_dtypes[input_idx]
            )
        else:
            try:
                sample_input = numpy.random.randint(
                    -1,
                    args.vocab_size + 1,
                    size=input_shapes[input_idx],
                    dtype=input_dtypes[input_idx],
                )
            except Exception:
                sample_input = numpy.random.rand(*input_shapes[input_idx]).astype(
                    input_dtypes[input_idx]
                )
        return numpy.ascontiguousarray(sample_input)

    for _ in range(args.num_warmup_iterations + args.num_iterations):
        sample_inputs = [
            _generate_sample_input(inp_idx) for inp_idx in range(len(input_shapes))
        ]
        yield sample_inputs
        del sample_inputs


def _run_model(args, model: Any, batch: List[numpy.ndarray]) -> List[numpy.ndarray]:
    outputs = None
    if args.engine == ORT_ENGINE:
        outputs = model.run(
            [out.name for out in model.get_outputs()],  # outputs
            {model.get_inputs()[0].name: batch},  # inputs dict
        )
    else:  # deepsparse
        outputs = model.run([batch])
    return outputs


def main():
    _config = parse_args()
    benchmark(args=_config)


if __name__ == "__main__":
    main()
