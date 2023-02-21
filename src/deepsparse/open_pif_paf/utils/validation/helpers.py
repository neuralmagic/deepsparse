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
import logging
from typing import List

import torch
from deepsparse.open_pif_paf.schemas import OpenPifPafFields
from openpifpaf import (
    Predictor,
    __version__,
    datasets,
    decoder,
    logger,
    network,
    show,
    transforms,
    visualizer,
)
from openpifpaf.eval import CustomFormatter, Evaluator


LOG = logging.getLogger(__name__)

__all__ = ["cli", "apply_deepsparse_preprocessing", "deepsparse_fields_to_torch"]


# adapted from OPENPIFPAF GITHUB:
# https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/eval.py
# the appropriate edits are marked with # deepsparse edit: <edit comment>
def cli():
    parser = argparse.ArgumentParser(
        prog="python3 -m openpifpaf.eval",
        usage="%(prog)s [options]",
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="OpenPifPaf {version}".format(version=__version__),
    )

    datasets.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    Predictor.cli(parser, skip_batch_size=True, skip_loader_workers=True)
    show.cli(parser)
    visualizer.cli(parser)
    Evaluator.cli(parser)

    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")
    parser.add_argument(
        "--output", default=None, help="output filename without file extension"
    )
    parser.add_argument(
        "--watch",
        default=False,
        const=60,
        nargs="?",
        type=int,
        help=(
            "Watch a directory for new checkpoint files. "
            "Optionally specify the number of seconds between checks."
        ),
    )

    # deepsparse edit: replace the parse_args call with parse_known_args
    args, unknown = parser.parse_known_args()

    # add args.device
    args.device = torch.device("cpu")
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.pin_memory = True
    LOG.debug("neural network device: %s", args.device)

    datasets.configure(args)
    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    visualizer.configure(args)
    Evaluator.configure(args)

    return args


def deepsparse_fields_to_torch(
    fields_batch: OpenPifPafFields, device="cpu"
) -> List[List[torch.Tensor]]:
    """
    Convert a batch of fields from the deepsparse
    openpifpaf fields schema to torch tensors

    :param fields_batch: the batch of fields to convert
    :param device: the device to move the tensors to
    :return: a list of lists of torch tensors. The first
        list is the batch dimension, the second list
        contains two tensors: Cif and Caf field values
    """
    return [
        [
            torch.from_numpy(array).to(device)
            for field in fields_batch.fields
            for array in field
        ]
    ]


def apply_deepsparse_preprocessing(
    data_loader: torch.utils.data.DataLoader, img_size: int
) -> torch.utils.data.DataLoader:
    """
    Replace the CenterPadTight transform in the data loader
    with a CenterPad transform to ensure that the images
    from the data loader are (B, 3, D, D) where D is
    the img_size. This function changes `data_loader`
    in place

    :param data_loader: the data loader to modify
    :param img_size: the image size to pad to
    """
    data_loader.dataset.preprocess.preprocess_list[2] = transforms.CenterPad(img_size)
