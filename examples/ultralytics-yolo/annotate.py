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
Annotation script for running YOLO models using DeepSparse and other inference engines.
Supports .jpg images, .mp4 movies, and webcam streaming.

##########
Command help:
usage: annotate.py [-h] --source SOURCE [-e {deepsparse,onnxruntime,torch}]
                   [--image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]]
                   [-c NUM_CORES] [-s NUM_SOCKETS] [-q] [--fp16]
                   [--device DEVICE] [--save-dir SAVE_DIR] [--name NAME]
                   [--target-fps TARGET_FPS] [--no-save]
                   [--model-config MODEL_CONFIG]
                   model_filepath

Annotate images, videos, and streams with sparsified or non-sparsified YOLO models

positional arguments:
  model_filepath        The full file path of the ONNX model file or SparseZoo
                        stub to the model for DeepSparse and ONNX Runtime
                        Engines. Path to a .pt loadable PyTorch Module for
                        torch - the Module can be the top-level object loaded
                        or loaded into 'model' in a state dict

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       File path to image or directory of .jpg files, a .mp4
                        video, or an integer (i.e. 0) for webcam
  -e {deepsparse,onnxruntime,torch}, --engine {deepsparse,onnxruntime,torch}
                        Inference engine backend to run on. Choices are
                        'deepsparse', 'onnxruntime', and 'torch'. Default is
                        'deepsparse'
  --image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Image shape to run model with, must be two integers.
                        Default is 640 640
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the annotations
                        with, defaults to None where it uses all physical
                        cores available on the system. For DeepSparse
                        benchmarks, this value is the number of cores per
                        socket
  -s NUM_SOCKETS, --num-sockets NUM_SOCKETS
                        For DeepSparse Engine only. The number of physical
                        cores to run the annotations. Defaults to None where
                        it uses all sockets available on the system
  -q, --quantized-inputs
                        Set flag to execute with int8 inputs instead of
                        float32
  --fp16                Set flag to execute with torch in half precision
                        (fp16)
  --device DEVICE       Torch device id to run the model with. Default is cpu.
                        Non-cpu only supported for Torch benchmarking. Default
                        is 'cpu' unless running with Torch and CUDA is
                        available, then cuda on device 0. i.e. 'cuda', 'cpu',
                        0, 'cuda:1'
  --save-dir SAVE_DIR   directory to save all results to. defaults to
                        'annotation_results'
  --name NAME           name of directory in save-dir to write results to.
                        defaults to {engine}-annotations-{run_number}
  --target-fps TARGET_FPS
                        target FPS when writing video files. Frames will be
                        dropped to closely match target FPS. --source must be
                        a video file and if target-fps is greater than the
                        source video fps then it will be ignored. Default is
                        None
  --no-save             set flag when source is from webcam to not save
                        results. not supported for non-webcam sources
  --model-config MODEL_CONFIG
                        YOLO config YAML file to override default anchor
                        points when post-processing. Defaults to use standard
                        YOLOv3/YOLOv5 anchors

##########
Example command for running webcam annotations with pruned quantized YOLOv3:
python annotate.py \
    zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
    --source 0 \
    --quantized-inputs \
    --image-shape 416 416

##########
Example command for running video annotations with pruned YOLOv5l:
python annotate.py \
    zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98 \
    --source my_video.mp4 \
    --image-shape 416 416

##########
Example command for running image annotations with using PyTorch CPU YOLOv3:
python annotate.py \
    path/to/yolo-v3.pt \
    --source path/to/my/jpg/images \
    --device cpu \
    --image-shape 416 416
"""


import argparse
import logging
import os
import time
from typing import Any, List, Union

import numpy
import onnx
import onnxruntime

import cv2
import torch
from deepsparse import compile_model
from deepsparse_utils import (
    YoloPostprocessor,
    annotate_image,
    get_yolo_loader_and_saver,
    modify_yolo_onnx_input_shape,
    postprocess_nms,
    yolo_onnx_has_postprocessing,
)
from sparseml.onnx.utils import override_model_batch_size


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"

_LOGGER = logging.getLogger(__name__)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Annotate images, videos, and streams with sparsified YOLO models"
    )

    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full file path of the ONNX model file or SparseZoo stub to the model "
            "for DeepSparse and ONNX Runtime Engines. Path to a .pt loadable PyTorch "
            "Module for torch - the Module can be the top-level object "
            "loaded or loaded into 'model' in a state dict"
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "File path to image or directory of .jpg files, a .mp4 video, "
            "or an integer (i.e. 0) for webcam"
        ),
    )
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[DEEPSPARSE_ENGINE, ORT_ENGINE, TORCH_ENGINE],
        help=(
            "Inference engine backend to run on. Choices are 'deepsparse', "
            "'onnxruntime', and 'torch'. Default is 'deepsparse'"
        ),
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        default=(640, 640),
        nargs="+",
        help="Image shape to run model with, must be two integers. Default is 640 640",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the annotations with, "
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
            "For DeepSparse Engine only. The number of physical cores to run the "
            "annotations. Defaults to None where it uses all sockets available on the "
            "system"
        ),
    )
    parser.add_argument(
        "-q",
        "--quantized-inputs",
        help=("Set flag to execute with int8 inputs instead of float32"),
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        help=("Set flag to execute with torch in half precision (fp16)"),
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=_parse_device,
        default=None,
        help=(
            "Torch device id to run the model with. Default is cpu. Non-cpu "
            " only supported for Torch benchmarking. Default is 'cpu' "
            "unless running with Torch and CUDA is available, then CUDA on "
            "device 0. i.e. 'cuda', 'cpu', 0, 'cuda:1'"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="annotation_results",
        help="directory to save all results to. defaults to 'annotation_results'",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "name of directory in save-dir to write results to. defaults to "
            "{engine}-annotations-{run_number}"
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "target FPS when writing video files. Frames will be dropped to "
            "closely match target FPS. --source must be a video file and if target-fps "
            "is greater than the source video fps then it will be ignored. Default is "
            "None"
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "set flag when source is from webcam to not save results. not supported "
            "for non-webcam sources"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help=(
            "YOLO config YAML file to override default anchor points when "
            "post-processing. Defaults to use standard YOLOv3/YOLOv5 anchors"
        ),
    )

    args = parser.parse_args(args=arguments)
    if args.engine == TORCH_ENGINE and args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return args


def _parse_device(device: Union[str, int]) -> Union[str, int]:
    try:
        return int(device)
    except Exception:
        return device


def _get_save_dir(args) -> str:
    name = args.name or f"{args.engine}-annotations"
    save_dir = os.path.join(args.save_dir, name)
    idx = 2
    while os.path.exists(save_dir):
        save_dir = os.path.join(args.save_dir, f"{name}-{idx}")
        idx += 1
    _LOGGER.info(f"Results will be saved to {save_dir}")
    return save_dir


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
        args.model_filepath, _ = modify_yolo_onnx_input_shape(
            args.model_filepath, args.image_shape
        )
        has_postprocessing = yolo_onnx_has_postprocessing(args.model_filepath)

    # load model
    if args.engine == DEEPSPARSE_ENGINE:
        _LOGGER.info(f"Compiling DeepSparse model for {args.model_filepath}")
        model = compile_model(args.model_filepath, 1, args.num_cores, args.num_sockets)
        if args.quantized_inputs and not model.cpu_vnni:
            _LOGGER.warning(
                "WARNING: VNNI instructions not detected, "
                "quantization speedup not well supported"
            )
    elif args.engine == ORT_ENGINE:
        _LOGGER.info(f"Loading onnxruntime model for {args.model_filepath}")

        sess_options = onnxruntime.SessionOptions()
        if args.num_cores is not None:
            sess_options.intra_op_num_threads = args.num_cores
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        onnx_model = onnx.load(args.model_filepath)
        override_model_batch_size(onnx_model, 1)
        model = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )
    elif args.engine == TORCH_ENGINE:
        _LOGGER.info(f"Loading torch model for {args.model_filepath}")
        model = torch.load(args.model_filepath)
        if isinstance(model, dict):
            model = model["model"]
        model.to(args.device)
        model.eval()
        if args.fp16:
            _LOGGER.info("Using half precision")
            model.half()
        else:
            _LOGGER.info("Using full precision")
            model.float()
        has_postprocessing = True

    return model, has_postprocessing


def _preprocess_batch(args, batch: numpy.ndarray) -> Union[numpy.ndarray, torch.Tensor]:
    if len(batch.shape) == 3:
        batch = batch.reshape(1, *batch.shape)
    if args.engine == TORCH_ENGINE:
        batch = torch.from_numpy(batch.copy())
        batch = batch.to(args.device)
        batch = batch.half() if args.fp16 else batch.float()
        batch /= 255.0
    else:
        if not args.quantized_inputs:
            batch = batch.astype(numpy.float32) / 255.0
        batch = numpy.ascontiguousarray(batch)
    return batch


def _run_model(
    args, model: Any, batch: Union[numpy.ndarray, torch.Tensor]
) -> List[Union[numpy.ndarray, torch.Tensor]]:
    outputs = None
    if args.engine == TORCH_ENGINE:
        outputs = model(batch)
    elif args.engine == ORT_ENGINE:
        outputs = model.run(
            [out.name for out in model.get_outputs()],  # outputs
            {model.get_inputs()[0].name: batch},  # inputs dict
        )
    else:  # deepsparse
        outputs = model.run([batch])
    return outputs


def annotate(args):
    save_dir = _get_save_dir(args)
    model, has_postprocessing = _load_model(args)
    loader, saver, is_video = get_yolo_loader_and_saver(
        args.source, save_dir, args.image_shape, args
    )
    is_webcam = args.source.isnumeric()

    postprocessor = (
        YoloPostprocessor(args.image_shape, args.model_config)
        if not has_postprocessing
        else None
    )

    for iteration, (inp, source_img) in enumerate(loader):
        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()
        iter_start = time.time()

        # pre-processing
        batch = _preprocess_batch(args, inp)

        # inference
        outputs = _run_model(args, model, batch)

        # post-processing
        if postprocessor:
            outputs = postprocessor.pre_nms_postprocess(outputs)
        else:
            outputs = outputs[0]  # post-processed values stored in first output

        # NMS
        outputs = postprocess_nms(outputs)[0]

        if args.device not in ["cpu", None]:
            torch.cuda.synchronize()

        # annotate
        measured_fps = (
            args.target_fps or (1.0 / (time.time() - iter_start)) if is_video else None
        )
        annotated_img = annotate_image(
            source_img,
            outputs,
            model_input_size=args.image_shape,
            images_per_sec=measured_fps,
        )

        # display
        if is_webcam:
            cv2.imshow("annotations", annotated_img)
            cv2.waitKey(1)

        # save
        if saver:
            saver.save_frame(annotated_img)

        iter_end = time.time()
        elapsed_time = 1000 * (iter_end - iter_start)
        _LOGGER.info(f"Inference {iteration} processed in {elapsed_time} ms")

    if saver:
        saver.close()
    _LOGGER.info(f"Results saved to {save_dir}")


def main():
    args = parse_args()
    assert len(args.image_shape) == 2
    args.image_shape = tuple(args.image_shape)

    annotate(args)


if __name__ == "__main__":
    main()
