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

import torch
from deepsparse.open_pif_paf.utils.validation.deepsparse_evaluator import (
    DeepSparseEvaluator,
)
from deepsparse.open_pif_paf.utils.validation.deepsparse_predictor import (
    DeepSparsePredictor,
)
from openpifpaf import __version__, datasets, decoder, logger, network, show, visualizer
from openpifpaf.eval import CustomFormatter


LOG = logging.getLogger(__name__)

__all__ = ["cli"]


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
    DeepSparsePredictor.cli(parser, skip_batch_size=True, skip_loader_workers=True)
    show.cli(parser)
    visualizer.cli(parser)
    DeepSparseEvaluator.cli(parser)

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
    args, _ = parser.parse_known_args()

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
    DeepSparsePredictor.configure(args)
    show.configure(args)
    visualizer.configure(args)
    DeepSparseEvaluator.configure(args)

    return args
