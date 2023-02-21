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

from deepsparse import Pipeline
from deepsparse.open_pif_paf.utils.validation import DeepSparseEvaluator, cli


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
SUPPORTED_DATASET_CONFIGS = ["cocokp"]

logging.basicConfig(level=logging.INFO)


@click.command(
    context_settings=(
        dict(token_normalize_func=lambda x: x.replace("-", "_"), show_default=True)
    )
)
@click.option(
    "--model-path",
    required=True,
    help="Path to the OpenPifPaf onnx model or" "SparseZoo stub to be evaluated.",
)
@click.option(
    "--dataset",
    type=str,
    default="cocokp",
    show_default=True,
    help="Dataset name supported by the openpifpaf framework. ",
)
@click.option(
    "--num-cores",
    type=int,
    default=None,
    show_default=True,
    help="Number of CPU cores to run deepsparse with, default is all available",
)
@click.option(
    "--image_size",
    type=int,
    default=641,
    show_default=True,
    help="Image size to use for evaluation. Will "
    "be used to resize images to the same size "
    "(B, C, image_size, image_size)",
)
@click.option(
    "--name-validation-run",
    type=str,
    default="openpifpaf_validation",
    show_default=True,
    help="Name of the validation run, used for" "creating a file to store the results",
)
@click.option(
    "--engine-type",
    default=DEEPSPARSE_ENGINE,
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE]),
    show_default=True,
    help="engine type to use, valid choices: ['deepsparse', 'onnxruntime']",
)
@click.option(
    "--device",
    default="cuda",
    type=str,
    show_default=True,
    help="Use 'device=cpu' or pass valid CUDA device(s) if available, "
    "i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU",
)
def main(
    model_path: str,
    dataset: str,
    num_cores: Optional[int],
    image_size: int,
    name_validation_run: str,
    engine_type: str,
    device: str,
):

    if dataset not in SUPPORTED_DATASET_CONFIGS:
        raise ValueError(
            f"Dataset {dataset} is not supported. "
            f"Supported datasets are {SUPPORTED_DATASET_CONFIGS}"
        )
    args = cli()
    args.dataset = dataset
    args.output = name_validation_run
    args.device = device

    if dataset == "cocokp":
        # eval for coco keypoints dataset
        args.coco_eval_long_edge = image_size

    pipeline = Pipeline.create(
        task="open_pif_paf",
        model_path=model_path,
        engine_type=engine_type,
        num_cores=num_cores,
        image_size=image_size,
        return_cifcaf_fields=True,
    )

    evaluator = DeepSparseEvaluator(
        pipeline=pipeline,
        dataset_name=args.dataset,
        skip_epoch0=False,
        img_size=image_size,
    )
    if args.watch:
        # this pathway has not been tested
        # and is not supported
        assert args.output is None
        evaluator.watch(args.checkpoint, args.watch)
    else:
        evaluator.evaluate(args.output)


if __name__ == "__main__":
    main()
