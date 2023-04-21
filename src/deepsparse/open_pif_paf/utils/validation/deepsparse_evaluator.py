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

import json
import logging
import os
import sys
import typing as t
from collections import defaultdict

from deepsparse import Pipeline
from deepsparse.open_pif_paf.utils.validation.deepsparse_predictor import (
    DeepSparsePredictor,
)
from deepsparse.open_pif_paf.utils.validation.helpers import (
    apply_deepsparse_preprocessing,
)
from openpifpaf import __version__, network
from openpifpaf.eval import Evaluator


LOG = logging.getLogger(__name__)

__all__ = ["DeepSparseEvaluator"]


# adapted from OPENPIFPAF GITHUB:
# https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/eval.py
# the appropriate edits are marked with # deepsparse edit: <edit comment>
class DeepSparseEvaluator(Evaluator):
    # deepsparse edit: allow for passing in a pipeline
    def __init__(self, pipeline: Pipeline, img_size: int, **kwargs):
        self.pipeline = pipeline
        super().__init__(**kwargs)
        # deepsparse edit: required to enforce square images
        apply_deepsparse_preprocessing(self.data_loader, img_size)

    def evaluate(self, output: t.Optional[str]):
        # generate a default output filename
        if output is None:
            assert self.args is not None
            output = self.default_output_name(self.args)

        # skip existing?
        if self.skip_epoch0:
            assert network.Factory.checkpoint is not None
            if network.Factory.checkpoint.endswith(".epoch000"):
                print("Not evaluating epoch 0.")
                return
        if self.skip_existing:
            stats_file = output + ".stats.json"
            if os.path.exists(stats_file):
                print("Output file {} exists already. Exiting.".format(stats_file))
                return
            print(
                "{} not found. Processing: {}".format(
                    stats_file, network.Factory.checkpoint
                )
            )

        # deepsparse edit: allow for passing in a pipeline
        predictor = DeepSparsePredictor(
            pipeline=self.pipeline, head_metas=self.datamodule.head_metas
        )
        metrics = self.datamodule.metrics()

        total_time = self.accumulate(predictor, metrics)

        # model stats
        # deepsparse edit: removed model stats that are
        # only applicable to torch models

        # write
        additional_data = {
            "args": sys.argv,
            "version": __version__,
            "dataset": self.dataset_name,
            "total_time": total_time,
            "n_images": predictor.total_images,
            "decoder_time": predictor.total_decoder_time,
            "nn_time": predictor.total_nn_time,
        }

        metric_stats = defaultdict(list)
        for metric in metrics:
            if self.write_predictions:
                metric.write_predictions(output, additional_data=additional_data)

            this_metric_stats = metric.stats()
            assert len(this_metric_stats.get("text_labels", [])) == len(
                this_metric_stats.get("stats", [])
            )

            for k, v in this_metric_stats.items():
                metric_stats[k] = metric_stats[k] + v

        stats = dict(**metric_stats, **additional_data)

        # write stats file
        with open(output + ".stats.json", "w") as f:
            json.dump(stats, f)

        LOG.info("stats:\n%s", json.dumps(stats, indent=4))
        LOG.info(
            "time per image: decoder = %.0fms, nn = %.0fms, total = %.0fms",
            1000 * stats["decoder_time"] / stats["n_images"],
            1000 * stats["nn_time"] / stats["n_images"],
            1000 * stats["total_time"] / stats["n_images"],
        )
