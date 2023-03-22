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

import copy
import logging

import click

import pandas as pd
from sparsezoo.analyze import ModelAnalysis
from sparsezoo.analyze.cli import analyze_options, analyze_performance_options


LOGGER = logging.getLogger()


@click.command()
@analyze_options
@analyze_performance_options
def main(
    model_path: str,
    save: str,
    **kwargs,
):
    """
    DeepSparse Performance analysis for ONNX models.

    MODEL_PATH: can be a SparseZoo stub, or local path to a
    deployment-directory or ONNX model

    Examples:

    - Run model analysis on resnet

        deepsparse.analyze ~/models/resnet50.onnx
    """
    logging.basicConfig(level=logging.INFO)

    for unimplemented_feat in (
        "compare",
        "by_layer",
        "by_types",
        "save_graphs",
        "impose",
    ):
        if kwargs.get(unimplemented_feat):
            raise NotImplementedError(
                f"--{unimplemented_feat} has not been implemented yet"
            )

    LOGGER.info("Starting Analysis ...")
    analysis = ModelAnalysis.create(model_path)
    LOGGER.info("Analysis complete, collating results...")
    summary = analysis.summary()

    summary["MODEL"] = model_path
    _display_summary_as_table(summary)

    if save:
        LOGGER.info(f"Writing results to {save}")
        analysis.yaml(file_path=save)


def _display_summary_as_table(summary):
    summary_copy = copy.copy(summary)
    print(f"MODEL: {summary_copy.pop('MODEL')}", end="\n\n")
    footer = summary_copy.pop("Summary")

    for section_name, section_dict in summary_copy.items():
        print(f"{section_name.upper()}:")
        print(pd.DataFrame(section_dict).T.to_string(), end="\n\n")

    print("SUMMARY:")
    for footer_key, footer_value in footer.items():
        print(f"{footer_key}: {footer_value}")


if __name__ == "__main__":
    main()
