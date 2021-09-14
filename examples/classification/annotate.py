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
Annotation script for running Image classification models using DeepSparse and
other inference engines.
Supports .jpg images, .mp4 movies, and webcam streaming.

usage: annotate.py [-h] --source SOURCE [-e {deepsparse,onnxruntime,torch}]
                   [-i IMAGE_SHAPE [IMAGE_SHAPE ...]] [-c NUM_CORES]
                   [-s NUM_SOCKETS] [--fp16] [--device DEVICE]
                   [--save-dir SAVE_DIR] [--name NAME] [--no-save]
                   [-b BATCH_SIZE] [--target-fps TARGET_FPS]
                   model_filepath

Script for running Image classification models and annotating results using
the DeepSparse and other runtime engines such as PyTorch and onnxruntime

positional arguments:
  model_filepath        The full file source of the ONNX model file or
                        SparseZoo stub to the model for DeepSparse and ONNX
                        Runtime Engine.Visit
                        https://sparsezoo.neuralmagic.com/ to search model
                        stubs.Path to a .pt loadable PyTorch Module for torch
                        -Module can be the top-level object loaded or loaded
                        into 'model'in a state dict

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       File source to image or directory of .jpg files, a
                        .mp4 video, or an integer (i.e. 0) for webcam
  -e {deepsparse,onnxruntime,torch}, --engine {deepsparse,onnxruntime,torch}
                        Inference engine backend to run on. Choices are
                        ['deepsparse', 'onnxruntime', 'torch']. Default is
                        deepsparse
  -i IMAGE_SHAPE [IMAGE_SHAPE ...], --image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Image shape to run model with, must be two integers.
                        Default is (224, 224)
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the annotations
                        with, defaults to None where it uses all physical
                        cores available on the system. For DeepSparse
                        benchmarks, this value is the number of cores per
                        socket.
  -s NUM_SOCKETS, --num-sockets NUM_SOCKETS
                        For DeepSparse Engine only. The number of physical
                        cores to run the annotations. Defaults to None where
                        it uses all sockets available on the system
  --fp16                Set flag to execute with torch in half precision
                        (fp16)
  --device DEVICE       Torch device id to run the model with. Default is cpu.
                        Non-cpu only supported for Torch benchmarking. Default
                        is 'cpu' unless running with Torch and CUDA is
                        available, then CUDA on device 0. i.e. 'cuda', 'cpu',
                        0, 'cuda:1'
  --save-dir SAVE_DIR   directory to save all results to. defaults to
                        'annotation_results'
  --name NAME           name of directory in save-dir to write results to.
                        defaults to {engine}-annotations-{run_number}
  --no-save             set flag when source is from webcam to not save
                        results. not supported for non-webcam sources
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to use while annotating images
  --target-fps TARGET_FPS
                        target FPS when writing video files. Frames will be
                        dropped to closely match target FPS. --source must be
                        a video file and if target-fps is greater than the
                        source video fps then it will be ignored. Default is
                        None

##########
Example command for running video annotations with base resnet:
python annotate.py \
    zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none \
    --source my_video.mp4 \
    --image-shape 224 224

##########
Example command for running webcam annotations with base resnet:
python annotate.py \
    zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none \
    --source 0 \
    --image-shape 224 224

##########
Example command for running Image annotations with base resnet:
python annotate.py \
    zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none \
    --source /path/to/JPEGS/ \
    --image-shape 224 224
"""
import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy
import onnx
import onnxruntime

import cv2
import torch
import utils
from deepsparse import compile_model
from sparseml.onnx.utils import override_model_batch_size


_LOGGER = logging.getLogger(__name__)
ZOO_STUB_PREFIX = "zoo:"


@dataclass
class _Config:
    model_filepath: str
    source: Union[str, int]
    engine: str
    image_shape: Tuple[int, int]
    num_cores: Optional[int]
    num_sockets: Optional[int]
    fp16: bool
    device: Optional[Union[str, int]]
    save_dir: str
    name: str
    no_save: bool
    batch_size: int
    target_fps: Optional[float]

    def _validate_configuration(self):
        if not utils.is_valid_stub_or_existing_file(self.model_filepath):
            raise ValueError(
                f"model-filepath:{self.model_filepath} is NOT a valid " f"stub/source"
            )
        if self.source.isnumeric():
            self.source = int(self.source)
        utils.validate_image_source(self.source)
        self.engine = utils.Engine(self.engine)

        if len(self.image_shape) != 2:
            raise ValueError(
                f"Image shape must be 2 integers given: {self.image_shape}"
            )
        if self.device not in [None, "cpu"] and self.engine != utils.Engine:
            raise ValueError(f"device {self.device} is not supported for {self.engine}")
        if self.fp16 and self.engine != utils.Engine.Torch:
            raise ValueError(f"half precision is not supported for {self.engine}")
        if self.num_cores is not None and self.engine == utils.Engine.Torch:
            raise ValueError(
                f"overriding default num_cores not supported for {self.engine}"
            )
        if (
            self.num_cores is not None
            and self.engine == utils.Engine.ONNXRUNTIME
            and onnxruntime.__version__ < "1.7"
        ):
            raise ValueError(
                "overriding default num_cores not supported for onnxruntime < "
                "1.7.0. If using an older build with OpenMP, try setting the "
                "OMP_NUM_THREADS environment variable"
            )
        if self.num_sockets is not None and self.engine != utils.Engine.DEEPSPARSE:
            raise ValueError(
                f"Overriding num_sockets is not supported for {self.engine}"
            )
        if (
            self.engine == utils.Engine.DEEPSPARSE
            or self.engine == utils.Engine.ONNXRUNTIME
        ):
            self.model_filepath = utils.fix_onnx_input_shape(
                model_filepath=self.model_filepath,
                input_shape=self.image_shape,
            )
        if self.engine == utils.Engine.TORCH and self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.name = self.name or f"{self.engine.value}-annotations"
        assert self.batch_size >= 0, "batch size should be positive"
        self.save_dir = utils.get_save_dir_name(self.save_dir, self.name)

    def __post_init__(self):
        self._validate_configuration()


def _parse_args(args=Optional[List[str]]):
    parser = argparse.ArgumentParser(
        description="Script for running Image classification models and "
        "annotating results using the DeepSparse and other "
        "runtime engines such as PyTorch and onnxruntime"
    )

    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full file source of the ONNX model file or SparseZoo stub to "
            "the model for DeepSparse and ONNX Runtime Engine."
            "Visit https://sparsezoo.neuralmagic.com/ to search model stubs."
            "Path to a .pt loadable PyTorch  Module for torch -"
            "Module can be the top-level object loaded or loaded into 'model'"
            "in a state dict"
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "File source to image or directory of .jpg files, a .mp4 video, "
            "or an integer (i.e. 0) for webcam"
        ),
    )

    _default_engine = utils.Engine.DEEPSPARSE.value
    _supported_engines = utils.Engine.supported_engines()
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=_default_engine,
        choices=_supported_engines,
        help=(
            f"Inference engine backend to run on. Choices are "
            f"{_supported_engines}. Default is {_default_engine}"
        ),
    )

    _default_image_shape = (224, 224)
    parser.add_argument(
        "-i",
        "--image-shape",
        type=int,
        default=_default_image_shape,
        nargs="+",
        help="Image shape to run model with, must be two integers. "
        f"Default is {_default_image_shape}",
    )

    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=None,
        help=(
            "The number of physical cores to run the annotations with, "
            "defaults to None where it uses all physical cores available on "
            "the system. "
            "For DeepSparse benchmarks, this value is the number of cores "
            "per socket."
        ),
    )

    parser.add_argument(
        "-s",
        "--num-sockets",
        type=int,
        default=None,
        help=(
            "For DeepSparse Engine only. The number of physical cores to run "
            "the annotations. Defaults to None where it uses all sockets "
            "available on the system"
        ),
    )

    parser.add_argument(
        "--fp16",
        help="Set flag to execute with torch in half precision (fp16)",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        type=utils.parse_device,
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
        help="directory to save all results to. defaults to " "'annotation_results'",
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
        "--no-save",
        action="store_true",
        help=(
            "set flag when source is from webcam to not save results. not "
            "supported for non-webcam sources"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="The batch size to use while annotating images",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "target FPS when writing video files. Frames will be dropped to "
            "closely match target FPS. --source must be a video file and if "
            "target-fps "
            "is greater than the source video fps then it will be ignored. "
            "Default is "
            "None"
        ),
    )
    _args = parser.parse_args(args)
    _config = _Config(
        model_filepath=_args.model_filepath,
        source=_args.source,
        engine=_args.engine,
        image_shape=tuple(_args.image_shape),
        num_cores=_args.num_cores,
        num_sockets=_args.num_sockets,
        fp16=_args.fp16,
        device=_args.device,
        save_dir=_args.save_dir,
        name=_args.name,
        no_save=_args.no_save,
        batch_size=_args.batch_size,
        target_fps=_args.target_fps,
    )
    return _config


def _load_model(config: _Config):
    if config.engine == utils.Engine.DEEPSPARSE:
        _LOGGER.info(f"Loading DeepSparse Engine for {config.model_filepath}")

        model = compile_model(
            config.model_filepath,
            config.batch_size,
            config.num_cores,
            config.num_sockets,
        )

    elif config.engine == utils.Engine.ONNXRUNTIME:
        _LOGGER.info(f"Loading ONNX Runtime Engine for {config.model_filepath}")

        sess_options = onnxruntime.SessionOptions()
        if config.num_cores is not None:
            sess_options.intra_op_num_threads = config.num_cores

        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        onnx_model = onnx.load(config.model_filepath)
        override_model_batch_size(onnx_model, config.batch_size)
        model = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )
    elif config.engine == utils.Engine.TORCH:
        _LOGGER.info(f"Loading PyTorch Engine for {config.model_filepath}")

        model = torch.load(config.model_filepath)
        if isinstance(model, dict):
            model = model["model"]
        model.to(config.device)
        model.eval()

        if config.fp16:
            _LOGGER.info("Using half precision")
            model.half()
        else:
            _LOGGER.info("Using full precision")
            model.float()

    return model


def _preprocess_batch(
    config: _Config, batch: numpy.ndarray
) -> Union[numpy.ndarray, torch.Tensor]:
    single_image = len(batch.shape) == 3
    if single_image:
        batch = batch.reshape(1, *batch.shape)

    if config.engine == utils.Engine.TORCH:
        batch = torch.from_numpy(batch.copy())
        batch = batch.to(config.device)
        batch = batch.half() if config.fp16 else batch.float()
        batch /= 255.0
    else:
        batch = batch.astype(numpy.float32) / 255.0
        batch = numpy.ascontiguousarray(batch)

    return batch


def _run_model(
    config: _Config, model: Any, batch: Union[numpy.ndarray, torch.Tensor]
) -> List[Union[numpy.ndarray, torch.Tensor]]:
    if config.engine == utils.Engine.TORCH:
        outputs = model(batch)
    elif config.engine == utils.Engine.ONNXRUNTIME:
        model_outputs = [out.name for out in model.get_outputs()]
        model_inputs = {model.get_inputs()[0].name: batch}
        outputs = model.run(model_outputs, model_inputs)
    else:
        # DeepSparse
        outputs = model.run([batch])
    return outputs


def annotate(config: _Config):
    """
    Method to annotate images using Image classification models according to
    specified configuration

    :param config: _Config object with info for current annotation task
    """
    model = _load_model(config=config)
    _LOGGER.info(f"Loaded model: {model}")

    class_labels = utils.imagenet_labels_as_list()

    loader, saver, is_video = utils.get_loader_and_saver(
        config.source, config.save_dir, config.image_shape, config
    )
    is_webcam = isinstance(config.source, int)

    for batch_idx, (current_batch, source_images) in enumerate(loader):
        if config.device not in ["cpu", None]:
            torch.cuda.synchronize()

        batch = _preprocess_batch(config, current_batch)

        iter_start = time.time()

        # inference
        predictions = _run_model(config, model, batch)
        predictions = predictions[-1]

        elapsed_time = time.time() - iter_start
        measured_fps = config.batch_size / elapsed_time

        print("fps: ", measured_fps)

        predicted_labels = [
            class_labels[index]
            for index in numpy.argmax(numpy.asarray(predictions), axis=-1)
        ]

        for image, label in zip(source_images, predicted_labels):
            annotated_image = utils.annotate_image(
                image=image,
                annotation_text=f"class: {label} images/sec: {measured_fps}",
            )

            # display
            if is_webcam:
                cv2.imshow("annotations", annotated_image)
                cv2.waitKey(1)

            if saver:
                saver.save_frame(annotated_image)
                _LOGGER.info(f"Results saved to {config.save_dir}")

    if saver:
        saver.close()


def main(args=Optional[List[str]]):
    """
    Driver function

    :param args: Command line arguments to parse and drive annotation
    """
    _config = _parse_args(args=args)
    annotate(_config)
    _cleanup(filepath=_config.model_filepath)


def _cleanup(filepath):
    os.unlink(filepath)


if __name__ == "__main__":
    main(args=sys.argv[1:])
