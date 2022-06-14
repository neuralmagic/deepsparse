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


import logging
from typing import Optional

import click

import cv2
from deepsparse.image_classification.constants import IMAGENET_LABELS
from deepsparse.image_classification.utils import annotate as _annotate
from deepsparse.pipeline import Pipeline
from deepsparse.yolo.utils import (
    annotate,
    get_annotations_save_dir,
    get_yolo_loader_and_saver,
)
from deepsparse.yolo.utils.cli_helpers import create_dir_callback


ic_default_stub = (
    "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none"
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
    default=ic_default_stub,
    help="Path/SparseZoo stub to the model file to be used for annotation",
    show_default=True,
)
@click.option(
    "--source",
    type=str,
    required=True,
    help="File path to an image file OR an .mp4 video file OR a directory of "
    "with image files",
)
@click.option(
    "--engine",
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE, TORCH_ENGINE]),
    default=DEEPSPARSE_ENGINE,
    help="Inference engine backend to run on. Choices are 'deepsparse', "
    "'onnxruntime', and 'torch'. Default is 'deepsparse'",
)
@click.option(
    "--display_image_shape",
    "--display_image_shape",
    type=int,
    nargs=2,
    default=(640, 640),
    help="Image shape of the annotated image, must be two integers",
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
    "--display",
    "--display",
    is_flag=True,
    help="If set to True, annotated images using `imshow()` method.",
    show_default=True,
)
@click.option(
    "--no_save",
    "--no-save",
    is_flag=True,
    help="If set to True, annotated images are saved to disk",
    show_default=True,
)
def main(
    model_filepath: str,
    source: str,
    engine: str,
    display_image_shape: tuple,
    num_cores: Optional[int],
    save_dir: str,
    name: Optional[str],
    target_fps: Optional[float],
    no_save: bool,
    display: bool,
) -> None:
    """
    Annotation Script for Image Classification with DeepSparse
    """
    save_dir = get_annotations_save_dir(
        initial_save_dir=save_dir,
        tag=name,
        engine=engine,
    )

    loader, saver, is_video = get_yolo_loader_and_saver(
        path=source,
        save_dir=save_dir,
        image_shape=(224, 224),
        target_fps=target_fps,
        no_save=no_save,
    )

    cv_pipeline = Pipeline.create(
        task="image_classification",
        model_path=model_filepath,
        engine_type=engine,
        num_cores=num_cores,
        top_k=5,
        class_names={str(idx): label for idx, label in enumerate(IMAGENET_LABELS)},
    )

    for iteration, (input_image, source_image) in enumerate(loader):

        # annotate
        annotated_image = annotate(
            pipeline=cv_pipeline,
            annotation_func=_annotate,
            image_batch=input_image,
            original_images=[source_image],
            display_image_shape=display_image_shape,
        )

        if display:
            cv2.imshow("annotated", annotated_image)
            cv2.waitKey(1)

        # save
        if saver:
            saver.save_frame(annotated_image[0])

    if saver:
        saver.close()

    _LOGGER.info(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
