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
Helper methods and utilities
"""
import glob
import json
import logging
import os
import shutil
from enum import Enum, unique
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import numpy
import numpy as np
import onnx

import cv2
from sparseml.onnx.utils import get_tensor_dim_shape, set_tensor_dim_shape
from sparseml.utils import create_dirs
from sparsezoo import Zoo


ZOO_STUB_PREFIX = "zoo:"


@unique
class Engine(Enum):
    DEEPSPARSE = "deepsparse"
    ONNXRUNTIME = "onnxruntime"
    TORCH = "torch"

    @classmethod
    def supported_engines(cls):
        return [_engine.value for _engine in cls]


# helpers


def annotate_image(image: numpy.ndarray, annotation_text: str) -> numpy.ndarray:
    """
    Returns a new copy of image annotated with the given text, in the top
    left corner.

    :param image: A numpy array representing the image
    :param annotation_text: The text to annotate the image with
    :return: A copy of the image annotated with the given annotation text
    """
    image_copy = numpy.copy(image)

    (w, h), _ = cv2.getTextSize(annotation_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    x1, y1 = 10, 50
    image_copy = cv2.rectangle(image_copy, (x1, y1 - 20), (x1 + w, y1), (0, 0, 0), -1)
    image_copy = cv2.putText(
        image_copy,
        annotation_text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    return image_copy


def parse_device(device: Union[str, int]) -> Union[str, int]:
    """
    :param device: The device id to use for inference
    :return: An integer representing the device to run inference on or the
        string id if device id is not an integer value
    """
    try:
        return int(device)
    except Exception:
        return device


def get_save_dir_name(*args) -> str:
    """
    Returns a non-existing directory name from the given arguments

    :param args: directories to join together
    :return:
    """
    save_directory_name = joined_directory_name = os.path.join(*args)
    index = 2

    while Path(save_directory_name).exists():
        save_directory_name = os.path.join(f"{joined_directory_name}-{index}")
        index += 1
    print(f"Results will be saved to {save_directory_name}")
    return save_directory_name


def fix_onnx_input_shape(model_file_or_stub: str, input_shape: Tuple[int, int]) -> str:
    """
    Creates a new ONNX model from given model file or SparseZoo stub that
    accepts the input shape. If the given model already has the given input
    shape no modifications are made. Uses a tempfile to store the
    modified model file.

    :param model_file_or_stub: str SparseZoo stub or source to ONNX model file
    :param input_shape: Tuple of two integers representing the image shape to
        override ONNX model file with.
    :return: filepath to an onnx model with input reshaped to the
        given input_shape, will be the original source if the shape is the same.
    """
    given_model_path = model_path = model_file_or_stub
    if given_model_path.startswith("zoo:"):
        model = Zoo.load_model_from_stub(given_model_path)
        model_path = model.onnx_file.downloaded_path()
        logging.info(f"Downloaded {given_model_path} to {model_path}")

    model = onnx.load(model_path)
    model_input = model.graph.input[0]

    model_input_indexes = [2, 3]
    model_input_shape = tuple(
        get_tensor_dim_shape(model_input, index) for index in model_input_indexes
    )

    if model_input_shape == input_shape:
        return model_path  # No shape modification needed

    for input_index, dim_shape in zip(model_input_indexes, input_shape):
        set_tensor_dim_shape(model_input, input_index, dim_shape)

    tmp_file = NamedTemporaryFile(delete=False)
    onnx.save(model, tmp_file.name)

    logging.info(
        f"Overwrote original model input shape {model_input_shape} to "
        f"{input_shape}\n"
        f"Original model source: {given_model_path}, new temporary "
        f"model saved to {tmp_file.name}"
    )
    return tmp_file.name


# validators
def is_valid_stub_or_existing_file(stub_or_path: str) -> bool:
    """
    Utility method to test if given argument is either a valid SparseZoo stub
    or an existing file on the filesystem

    :param stub_or_path: string to test for validity
    :return: True if valid stub or existing file
    """
    is_valid_stub = (
        stub_or_path.startswith(ZOO_STUB_PREFIX) and stub_or_path.count("?") < 2
    )
    return is_valid_stub or Path(stub_or_path).is_file()


def is_valid_string_source(source: str) -> bool:
    """
    Utility function to test if string source is a valid image/video file or
    a non-empty directory of images directory

    :param source: str source to directory of images or a valid image/video file
    :return: bool True if valid string source else False
    """
    is_valid_file = Path(source).is_file() and (
        source.endswith(".jpeg") or source.endswith(".jpg") or source.endswith(".mp4")
    )
    is_valid_not_empty_dir = Path(source).is_dir() and (
        glob.glob(os.path.join(source, "*.jpeg"))
        or glob.glob(os.path.join(source, "*.jpg"))
    )
    return is_valid_file or is_valid_not_empty_dir


def is_valid_webcam(source: int) -> bool:
    """
    Utility function to test if webcam device is available and open
    follows “it's easier to ask for forgiveness than permission” idiom
    inspired from https://tinyurl.com/valid-webcam

    :param source: int representing the video camera device
    :return: bool True if valid video capture device else False
    """
    capture = cv2.VideoCapture(source)
    return capture is not None and capture.isOpened()


def validate_image_source(source: Union[str, int]) -> None:
    """
    Utility function to check if given source is either a valid file source,
    or a valid webcam

    :param source: String source to a valid file source, or an Integer
        representing a valid webcam
    """
    if isinstance(source, str) and not is_valid_string_source(source=source):
        raise ValueError(f"source:{source} is NOT a valid image source")
    if isinstance(source, int) and not is_valid_webcam(source=source):
        raise AssertionError(f"Unable to open video source:{source}")


def imagenet_labels_as_list() -> List[str]:
    """
    :return: ImageNet labels as a list
    """
    imagenet_json = "imagenet.json"
    with open(imagenet_json) as f:
        imagenet_data = json.load(f)
    return list(imagenet_data["label_mappings"].values())


def get_loader_and_saver(
    source: Union[str, int],
    save_dir: str,
    image_size: Tuple[int] = (224, 224),
    config: Any = None,
) -> Union[Iterable, Any, bool]:
    """
    :param source: file path to image or directory of .jpg files, a .mp4
    video, or an integer (i.e. 0) for web-cam
    :param save_dir: path of directory to save to
    :param image_size: size of input images to model
    :param config: configuration for annotation task
    :return: image loader iterable, result saver objects
        images, video, or web-cam based on path given, and a boolean value
        that is True is the returned objects load videos
    """

    source_is_webcam = isinstance(source, int)
    source_is_video = source.endswith(".mp4")
    is_video = source_is_webcam or source_is_video

    if source_is_webcam:
        loader = WebcamLoader(
            camera=source, image_size=image_size, batch_size=config.batch_size
        )
        saver = (
            None
            if config.no_save
            else VideoSaver(
                save_dir=save_dir,
                original_fps=30,
                output_frame_size=loader.original_frame_size,
                target_fps=config.target_fps,
            )
        )

    elif source_is_video:
        loader = VideoBatchLoader(
            path=source, image_size=image_size, batch_size=config.batch_size
        )
        saver = VideoSaver(
            save_dir=save_dir,
            original_fps=loader.original_fps,
            output_frame_size=loader.original_frame_size,
            target_fps=config.target_fps,
        )

    else:
        loader = ImageBatchLoader(
            path=source, image_size=image_size, batch_size=config.batch_size
        )
        saver = ImagesSaver(save_dir=save_dir)

    return loader, saver, is_video


# Loaders


def load_image(
    image: Union[str, numpy.ndarray], image_size: Tuple[int] = (224, 224)
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    :param image: file source to image or raw image array
    :param image_size: target shape for image
    :return: Image loaded into numpy and reshaped to the given shape and the
    original image
    """
    image = cv2.imread(image) if isinstance(image, str) else image
    img_resized = cv2.resize(image, image_size)
    img_transposed = img_resized[:, :, ::-1].transpose(2, 0, 1)
    return img_transposed, image


class ImageBatchLoader:
    """
    Class for pre-processing and iterating over images to be used as input
    for image classification models

    :param path: Filepath to single image file or directory of image files to
        load, glob paths also valid
    :param image_size: size of input images to model
    """

    def __init__(
        self, path: str, image_size: Tuple[int] = (224, 224), batch_size: int = 1
    ):
        self._path = path
        self._image_size = image_size
        self._batch_size = batch_size

        if Path(path).is_dir():
            self._image_file_paths = [
                os.path.join(path, file_name) for file_name in os.listdir(path)
            ]
        elif "*" in path:
            self._image_file_paths = glob.glob(path)
        elif Path(path).is_file():
            # single file
            self._image_file_paths = [path]
        else:
            raise ValueError(f"{path} is not a file, glob, or directory")

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        current_batch, images = [], []

        for image_path in self._image_file_paths:
            batch_image, image = load_image(
                image=image_path, image_size=self._image_size
            )
            current_batch.append(batch_image)
            images.append(image)

            if len(current_batch) == self._batch_size:
                yield np.asarray(current_batch), np.asarray(images)
                current_batch, images = [], []


class VideoBatchLoader:
    """
    Class for pre-processing and iterating over video frames to be used as
    input for Image classification models

    :param path: Filepath to single video file
    :param image_size: size of input images to model
    """

    def __init__(self, path: str, image_size: Tuple[int] = (224, 224), batch_size=1):
        self._path = path
        self._image_size = image_size
        self._vid = cv2.VideoCapture(self._path)
        self._total_frames = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._vid.get(cv2.CAP_PROP_FPS)
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        current_batch, images = [], []

        for _ in range(self._total_frames):
            loaded, frame = self._vid.read()
            if not loaded:
                break
            batch_image, image = load_image(image=frame, image_size=self._image_size)
            current_batch.append(batch_image)
            images.append(image)

            if len(current_batch) == self._batch_size:
                yield np.asarray(current_batch), np.asarray(images)
                current_batch, images = [], []
        self._vid.release()

    @property
    def original_fps(self) -> float:
        """
        :return: the frames per second of the video this object reads
        """
        return self._fps

    @property
    def original_frame_size(self) -> Tuple[int, int]:
        """
        :return: the original size of frames in the video this object reads
        """
        return (
            int(self._vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def total_frames(self) -> int:
        """
        :return: the total number of frames this object may laod from the video
        """
        return self._total_frames


class WebcamLoader:
    """
    Class for pre-processing and iterating over webcam frames to be used as
    input for Image classification models.

    Adapted from:
        https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py

    :param camera: Webcam index
    :param image_size: size of input images to model
    """

    def __init__(
        self, camera: int, image_size: Tuple[int, int] = (224, 224), batch_size: int = 1
    ):

        self._camera = camera
        self._image_size = image_size
        self._stream = cv2.VideoCapture(self._camera)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        current_batch, images = [], []

        while True:
            if cv2.waitKey(1) == ord("q"):  # q to quit
                self._stream.release()
                cv2.destroyAllWindows()
                break

            loaded, frame = self._stream.read()

            assert loaded, f"Could not load image from webcam {self._camera}"

            frame = cv2.flip(frame, 1)  # flip left-right
            batch_image, image = load_image(image=frame, image_size=self._image_size)
            current_batch.append(batch_image)
            images.append(image)

            if len(current_batch) == self._batch_size:
                yield np.asarray(current_batch), np.asarray(images)
                current_batch, images = [], []

    @property
    def original_frame_size(self) -> Tuple[int, int]:
        """
        :return: the original size of frames in the stream this object reads
        """
        return (
            int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )


# Savers


class ImagesSaver:
    """
    Base class for saving model outputs. Saves each image as an
    individual file in the given directory

    :param save_dir: source to directory to write to
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._idx = 0
        create_dirs(save_dir)

    def save_frame(self, image: numpy.ndarray):
        """
        :param image: numpy array of image to save
        """
        output_path = os.path.join(self._save_dir, f"result-{self._idx}.jpg")
        cv2.imwrite(output_path, image)
        self._idx += 1

    def close(self):
        """
        perform any clean-up tasks
        """
        pass


class VideoSaver(ImagesSaver):
    """
    Class for saving model outputs as a VideoFile

    :param save_dir: source to directory to write to
    :param original_fps: frames per second to save video with
    :param output_frame_size: size of frames to write
    :param target_fps: fps target for output video. if present, video
        will be written with a certain number of the original frames
        evenly dropped to match the target FPS.
    """

    def __init__(
        self,
        save_dir: str,
        original_fps: float,
        output_frame_size: Tuple[int, int],
        target_fps: Optional[float] = None,
    ):
        super().__init__(save_dir)

        self._output_frame_size = output_frame_size
        self._original_fps = original_fps

        if target_fps is not None and target_fps >= original_fps:
            print(
                f"target_fps {target_fps} is greater than source_fps "
                f"{original_fps}. target fps file will not be invoked"
            )
        self._target_fps = target_fps

        self._file_path = os.path.join(self._save_dir, "results.mp4")
        self._writer = cv2.VideoWriter(
            self._file_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            original_fps,
            self._output_frame_size,
        )
        self._n_frames = 0

    def save_frame(self, image: numpy.ndarray):
        """
        :param image: numpy array of image to save
        """
        self._writer.write(image)
        self._n_frames += 1

    def close(self):
        """
        perform any clean-up tasks
        """
        self._writer.release()
        if self._target_fps is not None and self._target_fps < self._original_fps:
            self._write_target_fps_video()

    def _write_target_fps_video(self):
        assert self._target_fps is not None
        num_frames_to_keep = int(
            self._n_frames * (self._target_fps / self._original_fps)
        )
        # adjust target fps so we can keep the same video duration
        adjusted_target_fps = num_frames_to_keep * (self._original_fps / self._n_frames)

        # select num_frames_to_keep evenly spaced frame idxs
        frame_indexes_to_keep = set(
            numpy.round(numpy.linspace(0, self._n_frames, num_frames_to_keep))
            .astype(int)
            .tolist()
        )

        # create new video writer for adjusted video
        vid_path = os.path.join(
            self._save_dir, f"_results-{adjusted_target_fps:.2f}fps.mp4"
        )
        fps_writer = cv2.VideoWriter(
            vid_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            adjusted_target_fps,
            self._output_frame_size,
        )

        # read from original video and write to FPS adjusted video
        saved_vid = cv2.VideoCapture(self._file_path)
        for idx in range(self._n_frames):
            _, frame = saved_vid.read()
            if idx in frame_indexes_to_keep:
                fps_writer.write(frame)

        saved_vid.release()
        fps_writer.release()
        shutil.move(vid_path, self._file_path)  # overwrite original file
