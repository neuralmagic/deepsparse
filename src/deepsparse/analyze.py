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
Script and utilities to run performance analysis on ONNX models with the
DeepSparse runtime engine
"""

import click


__all__ = [
    "ModelPerfAnalysis",
    "analyze",
]


class ModelPerfAnalysis(object):
    """
    A class to run and display performance analysis results on ONNX models
    with DeepSparse Engine
    """


def analyze() -> ModelPerfAnalysis:
    """
    Utility function to analyze a model's performance with the DeepSparse engine

    :return: A ModelPerfAnalysis object
    """
    return ModelPerfAnalysis()


@click.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"),
        show_default=True,
        ignore_unknown_options=True,
    )
)
@click.option(
    "--file",
    required=True,
    type=str,
    help="Path to an onnx model, deployment_directory (containing model.onnx), "
    "SparseZoo Stub or an analysis yaml file from a previous run",
)
def main(file: str):
    """
    deepsparse.analyze utility to run performance analysis on ONNX models with
    DeepSparse runtime engine
    """
    print(f"file = {file}")


if __name__ == "__main__":
    main()
