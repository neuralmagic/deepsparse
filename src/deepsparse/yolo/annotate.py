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
Usage: deepsparse.object_detection.annotate [OPTIONS]

  Annotation Script for YOLO with DeepSparse

Options:
  --model_filepath, --model-filepath TEXT
                                  Path/SparseZoo stub to the model file to be
                                  used for annotation  [default: zoo:cv/detect
                                  ion/yolov5-s/pytorch/ultralytics/coco/pruned
                                  -aggressive_96]
  --source TEXT                   File path to an image or directory of image
                                  files, a .mp4 video, or an integer (i.e. 0)
                                  for webcam  [required]
  --engine [deepsparse|onnxruntime|torch]
                                  Inference engine backend to run on. Choices
                                  are 'deepsparse', 'onnxruntime', and
                                  'torch'. Default is 'deepsparse'
  --image_shape, --image_shape INTEGER
                                  Image shape to use for inference, must be
                                  two integers  [default: 640, 640]
  --num_cores, --num-cores INTEGER
                                  The number of physical cores to run the
                                  annotations with, defaults to using all
                                  physical cores available on the system. For
                                  DeepSparse benchmarks, this value is the
                                  number of cores per socket
  --save_dir, --save-dir DIRECTORY
                                  The path to the directory for saving results
                                  [default: annotation-results]
  --name TEXT                     Name of directory in save-dir to write
                                  results to. defaults to
                                  {engine}-annotations-{run_number}
  --target_fps, --target-fps FLOAT
                                  Target FPS when writing video files. Frames
                                  will be dropped to closely match target FPS.
                                  --source must be a video file and if target-
                                  fps is greater than the source video fps
                                  then it will be ignored
  --no_save, --no-save            Set flag when source is from webcam to not
                                  save results.Not supported for non-webcam
                                  sources  [default: False]
  --help                          Show this message and exit

#######
Examples:

1) deepsparse.object_detection.annotate --source PATH/TO/IMAGE.jpg
2) deepsparse.object_detection.annotate --source PATH/TO/VIDEO.mp4
3) deepsparse.object_detection.annotate --source 0
4) deepsparse.object_detection.annotate --source PATH/TO/IMAGE_DIR
"""
import logging
from typing import Optional

import click

import cv2
from deepsparse.pipeline import Pipeline
from deepsparse.utils.annotate import (
    annotate,
    get_annotations_save_dir,
    get_image_loader_and_saver,
)
from deepsparse.yolo.utils import annotate_image
from deepsparse.yolo.utils.cli_helpers import create_dir_callback


yolo_v5_default_stub = (
    "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96"
)

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model_filepath",
    "--model-filepath",
    type=str,
    default=yolo_v5_default_stub,
    help="Path/SparseZoo stub to the model file to be used for annotation",
    show_default=True,
)
@click.option(
    "--source",
    type=str,
    required=True,
    help="File path to image or directory of .jpg files, a .mp4 video, "
    "or an integer (i.e. 0) for webcam",
)
@click.option(
    "--engine",
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE, TORCH_ENGINE]),
    default=DEEPSPARSE_ENGINE,
    help="Inference engine backend to run on. Choices are 'deepsparse', "
    "'onnxruntime', and 'torch'. Default is 'deepsparse'",
)
@click.option(
    "--image_shape",
    "--image-shape",
    type=int,
    nargs=2,
    default=(640, 640),
    help="Image shape to use for inference, must be two integers",
    show_default=True,
)
@click.option(
    "--num_cores",
    "--num-cores",
    type=int,
    default=None,
    help="The number of physical cores to run the annotations with, "
    "defaults to using all physical cores available on the system."
    " For DeepSparse benchmarks, this value is the number of cores "
    "per socket",
    show_default=True,
)
@click.option(
    "--save_dir",
    "--save-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default="annotation-results",
    callback=create_dir_callback,
    help="The path to the directory for saving results",
    show_default=True,
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Name of directory in save-dir to write results to. defaults to "
    "{engine}-annotations-{run_number}",
)
@click.option(
    "--target_fps",
    "--target-fps",
    type=float,
    default=None,
    help="Target FPS when writing video files. Frames will be dropped to "
    "closely match target FPS. --source must be a video file and if "
    "target-fps is greater than the source video fps then it "
    "will be ignored",
    show_default=True,
)
@click.option(
    "--no_save",
    "--no-save",
    is_flag=True,
    help="Set flag when source is from webcam to not save results."
    "Not supported for non-webcam sources",
    show_default=True,
)
def main(
    model_filepath: str,
    source: str,
    engine: str,
    image_shape: tuple,
    num_cores: Optional[int],
    save_dir: str,
    name: Optional[str],
    target_fps: Optional[float],
    no_save: bool,
) -> None:
    """
    Annotation Script for YOLO with DeepSparse
    """
    save_dir = get_annotations_save_dir(
        initial_save_dir=save_dir,
        tag=name,
        engine=engine,
    )

    loader, saver, is_video = get_image_loader_and_saver(
        path=source,
        save_dir=save_dir,
        image_shape=image_shape,
        target_fps=target_fps,
        no_save=no_save,
    )

    is_webcam = source.isnumeric()
    yolo_pipeline = Pipeline.create(
        task="yolo",
        model_path=model_filepath,
        class_names="coco",
        engine_type=engine,
        num_cores=num_cores,
    )

    for iteration, (input_image, source_image) in enumerate(loader):

        # annotate
        annotated_image = annotate(
            pipeline=yolo_pipeline,
            annotation_func=annotate_image,
            image=input_image,
            target_fps=target_fps,
            calc_fps=is_video,
            original_image=source_image,
            model_input_size=image_shape,
        )

        if is_webcam:
            cv2.imshow("annotated", annotated_image)
            cv2.waitKey(1)

        # save
        if saver:
            saver.save_frame(annotated_image)

    if saver:
        saver.close()

    _LOGGER.info(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
