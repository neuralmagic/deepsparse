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

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy
import onnx
import onnxruntime
from tqdm.auto import tqdm

import torch
from deepsparse import compile_model
from deepsparse.benchmark import BenchmarkResults
from sparseml.onnx.utils import override_model_batch_size
from sparsezoo.models.detection import yolo_v3 as zoo_yolo_v3
from sparsezoo.utils import load_numpy_list


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"


def load_random_data():
    """
    Load random tensors for benchmarking bert
    TODO
    """
    pass


def modify_bert_onnx_input_shape(model_filepath, image_shape):
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


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Benchmark sparsified YOLOv3 models")

    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full filepath of the ONNX model file or SparseZoo stub to the model "
            "for deepsparse and onnxruntime benchmarks. Path to a .pt loadable PyTorch "
            "Module for torch benchmarks - the Module can be the top-level object "
            "loaded or loaded into 'model' in a state dict"
        ),
    )

    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[DEEPSPARSE_ENGINE, ORT_ENGINE, TORCH_ENGINE],
        help=(
            "Inference engine backend to run benchmark_bert on. Choices are 'deepsparse', "
            "'onnxruntime', and 'torch'. Default is 'deepsparse'"
        ),
    )

    parser.add_argument(
        "--data-path",
        type=Optional[str],
        default=None,
        help=(
            "Optional filepath to image examples to run the benchmark_bert on. Can be path "
            "to directory, single image jpg file, or a glob path. All files should be "
            "in jpg format. If not provided, sample COCO images will be downloaded "
            "from the SparseZoo"
        ),
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to run the benchmark_bert for",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the benchmark_bert on, "
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
            "benchmark_bert on. Defaults to None where is uses all sockets available on the "
            "system"
        ),
    )
    parser.add_argument(
        "-i",
        "--num-iterations",
        help="The number of iterations the benchmark_bert will be run for",
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
        "-q",
        "--quantized-inputs",
        help="Set flag to execute benchmark_bert with int8 inputs instead of float32",
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        help="Set flag to execute torch benchmark_bert in half precision (fp16)",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=_parse_device,
        default=None,
        help=(
            "Torch device id to benchmark_bert the model with. Default is cpu. Non cpu "
            "benchmarking only supported for torch benchmarking. Default is 'cpu' "
            "unless running a torch benchmark_bert and cuda is available, then cuda on "
            "device 0. i.e. 'cuda', 'cpu', 0, 'cuda:1'"
        ),
    )

    args = parser.parse_args(args=args)
    if args.engine == TORCH_ENGINE and args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return args


def _parse_device(device: Union[str, int]) -> Union[str, int]:
    try:
        return int(device)
    except Exception:
        return device


def benchmark_bert(args):
    """
    Method to benchmark_bert inference times for BERT
    """
    model = _load_model(args)
    print("Loading dataset")
    dataset, _ = load_random_data(args.data_path, tuple(args.image_shape))
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
        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()
        iter_start = time.time()

        # inference
        outputs = _run_model(args, model, batch)

        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()
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
    if args.device not in [None, "cpu"] and args.engine != TORCH_ENGINE:
        raise ValueError(f"device {args.device} is not supported for {args.engine}")
    if args.fp16 and args.engine != TORCH_ENGINE:
        raise ValueError(f"half precision is not supported for {args.engine}")
    if args.quantized_inputs and args.engine == TORCH_ENGINE:
        raise ValueError(f"quantized inputs not supported for {args.engine}")
    if args.num_cores is not None and args.engine == TORCH_ENGINE:
        raise ValueError(
            f"overriding default num_cores not supported for {args.engine}"
        )
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
        args.model_filepath, _ = modify_bert_onnx_input_shape(
            args.model_filepath, args.image_shape
        )

    # load model
    if args.engine == DEEPSPARSE_ENGINE:
        print(f"Compiling deepsparse model for {args.model_filepath}")
        model = compile_model(
            args.model_filepath, args.batch_size, args.num_cores, args.num_sockets
        )
        if args.quantized_inputs and not model.cpu_vnni:
            print(
                "WARNING: VNNI instructions not detected, "
                "quantization speedup not well supported"
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
    elif args.engine == TORCH_ENGINE:
        print(f"loading torch model for {args.model_filepath}")
        model = torch.load(args.model_filepath)
        if isinstance(model, dict):
            model = model["model"]
        model.to(args.device)
        model.eval()
        if args.fp16:
            print("Using half precision")
            model.half()
        else:
            print("Using full precision")
    return model


def _run_model(
    args, model: Any, batch: Union[numpy.ndarray, torch.Tensor]
) -> List[Union[numpy.ndarray, torch.Tensor]]:
    outputs = None
    if args.engine == TORCH_ENGINE:
        outputs = model(batch)[0]
    elif args.engine == ORT_ENGINE:
        outputs = model.run(
            [out.name for out in model.get_outputs()],  # outputs
            {model.get_inputs()[0].name: batch},  # inputs dict
        )
    else:  # deepsparse
        outputs = model.run([batch])
    return outputs


def main(args: List[str] = None):
    _config = parse_args(args=args)
    benchmark_bert(args=_config)


if __name__ == "__main__":
    main(sys.argv[:1])
