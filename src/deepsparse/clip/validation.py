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

import warnings
from pathlib import Path
from typing import Union

import click
import numpy as np

from deepsparse.clip import (
    CLIPCaptionInput,
    CLIPTextInput,
    CLIPVisualInput,
    CLIPZeroShotInput,
)
from deepsparse.pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, BasePipeline


@click.command(
    context_settings=(
        dict(token_normalize_func=lambda x: x.replace("-", "_"), show_default=True)
    )
)
@click.option(
    "--visual-model",
    required=True,
    help="Path to the CLIP visual onnx model.",
)
@click.option(
    "--text-model",
    required=True,
    help="Path to the CLIP text onnx model.",
)
@click.option(
    "--decoder-model",
    required=False,
    help="Path to the CLIP decoder onnx model. Only required when task is 'caption'.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="The batch size to run predictions",
)
@click.option(
    "--engine-type",
    default=ORT_ENGINE,
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE]),
    show_default=True,
    help="engine type to use, valid choices: ['deepsparse', 'onnxruntime']",
)
@click.option(
    "--num-cores",
    type=int,
    default=None,
    show_default=True,
    help="Number of CPU cores to run deepsparse with, default is all available",
)
@click.option(
    "--images-file",
    type=str,
    default="sample_images.txt",
    show_default=True,
    help="Text file containing paths to images for zeroshot classification.",
)
@click.option(
    "--classes-file",
    type=str,
    default="sample_classes.txt",
    show_default=True,
    required=False,
    help="Text file containing a list of classes for zeroshot classification. "
    " Only required when task is 'zeroshot'",
)
@click.option(
    "--task",
    default="zeroshot",
    type=click.Choice(["zeroshot", "caption"]),
    show_default=True,
    help="CLIP task, options are: ['zeroshot', 'caption']",
)
def main(
    visual_model: Union[str, Path],
    text_model: Union[str, Path],
    decoder_model: Union[str, Path],
    batch_size: int,
    num_cores: int,
    engine_type: str,
    images_file: Union[str, Path],
    classes_file: Union[str, Path],
    task: str,
):
    engine_args = {
        "batch_size": batch_size,
        "num_cores": num_cores,
        "engine_type": engine_type,
    }

    with open(images_file) as f:
        images = f.readlines()
    images = [x.strip() for x in images]

    if task == "caption":
        if not decoder_model:
            raise ValueError(
                "For the captioning task, a decoder model must be provided"
            )
        if classes_file:
            warnings.warn(f"{classes_file} was provided but is not used for captioning")

        pipeline = BasePipeline.create(
            task="clip_caption",
            visual_model_path=visual_model,
            text_model_path=text_model,
            decoder_model_path=decoder_model,
            pipeline_engine_args=engine_args,
        )

        pipeline_input = CLIPCaptionInput(image=CLIPVisualInput(images=images))

        output = pipeline(pipeline_input).caption
        for i in range(len(images)):
            print(f"Class prediction for {images[i]}, {output[i]}")

    if task == "zeroshot":
        if not classes_file:
            raise ValueError(
                "For zeroshot classification a list of classes is required. Please use "
                " the 'classes-file' argument."
            )
        if decoder_model:
            warnings.warn(
                f"{decoder_model} was provided but it is not used for zeroshot "
                " classification"
            )

        pipeline = BasePipeline.create(
            task="clip_zeroshot",
            visual_model_path=visual_model,
            text_model_path=text_model,
            pipeline_engine_args=engine_args,
        )

        with open(classes_file) as f:
            classes = f.readlines()

        classes = [x.strip() for x in classes]

        pipeline_input = CLIPZeroShotInput(
            image=CLIPVisualInput(images=images), text=CLIPTextInput(text=classes)
        )

        output = pipeline(pipeline_input).text_scores
        for i in range(len(images)):
            print(f"Class prediction for {images[i]}, {classes[np.argmax(output[i])]}")


if __name__ == "__main__":
    main()
