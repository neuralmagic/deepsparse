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
######
Command help:
Usage: deepsparse.eval [OPTIONS]

  Module for evaluating models on the various evaluation integrations

Options:
    --target TARGET     A path to a remote or local directory containing ONNX/torch model
                        (including all the auxiliary files) or a SparseZoo stub
    -d DATASETS, --datasets DATASETS
                        The datasets to evaluate on. Can be a string for a single dataset
                        or a list of strings for multiple datasets
    -i INTEGRATION, --integration INTEGRATION
                        The name of the evaluation integration to use. Must be a valid
                        integration name that is registered in the evaluation registry
    -e ENGINE_TYPE, --engine_type ENGINE_TYPE
                        Inference engine to use for the evaluation. The default
                        is the DeepSparse engine. If the evaluation should be run
                        without initializing a pipeline (e.g. for the evaluation
                        of a torch model), the engine type should be set to None
    -s SAVE_PATH, --save_path SAVE_PATH
                        The path to save the evaluation results.
                        By default the results will be saved in the
                        current directory under the name 'result.[extension]'.
    -t TYPE_SERIALIZATION, --type_serialization TYPE_SERIALIZATION
                        The serialization type to use save the evaluation results.
                        The default is json
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to use for the evaluation. Must be greater than 0
    -metrics METRICS, --metrics METRICS
                        The name of the metrics to evaluate on. Can be a string for a single metric
                        or a list of strings for multiple metrics.
    -splits SPLITS, --splits SPLITS
                        The name of the splits to evaluate on. Can be a string for a single split
                        or a list of strings for multiple splits.

#########
EXAMPLES
#########

##########
Example command for evaluating a quantized MPT model from SparseZoo using the Deepsparse Engine.
The evaluation will be run using `lm-evaluation-harness` on `hellaswag` and `gsm8k` datasets:
deepsparse.eval zoo:mpt-7b-mpt_pretrain-base_quantized \
                --datasets hellaswag, gsm8k \
                --integration lm-evaluation-harness \

"""  # noqa: E501
import logging
from typing import List, Union

import click

from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.evaluation.results import Result, print_result, save_evaluations
from src.deepsparse.evaluation.utils import get_save_path
from src.deepsparse.pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE


_LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--target",
    type=click.Path(dir_okay=True, file_okay=True),
    required=True,
    help="A path to a remote or local directory containing ONNX/torch model "
    "(including all the auxiliary files) or a SparseZoo stub",
)
@click.option(
    "-d",
    "--datasets",
    type=click.UNPROCESSED,
    required=True,
    help="The datasets to evaluate on. Can be a string for a single dataset "
    "or a list of strings for multiple datasets",
)
@click.option(
    "-i",
    "--integration",
    type=str,
    required=True,
    help="The name of the evaluation integration to use. Must be a valid "
    "integration name that is registered in the evaluation registry",
)
@click.option(
    "-e",
    "--engine_type",
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE, None]),
    default=DEEPSPARSE_ENGINE,
    help="The engine to use for the evaluation. The default is the "
    "DeepSparse engine. If the evaluation should be run without "
    "initializing a pipeline (e.g. for the evaluation of a torch "
    "model), the engine type should be set to None",
)
@click.option(
    "-s",
    "--save_path",
    type=click.UNPROCESSED,
    default=None,
    help="The path to save the evaluation results. The results will "
    "be saved under the name 'result.yaml`/'result.json' depending on "
    "the serialization type. If argument is not provided, the results "
    "will be saved in the current directory",
)
@click.option(
    "-t",
    "--type_serialization",
    type=click.Choice(["yaml", "json"]),
    default="json",
    help="The serialization type to use save the evaluation results. "
    "The default is json",
)
@click.option(
    "-b",
    "--batch_size",
    type=int,
    default=1,
    help="The batch size to use for the evaluation. Must be greater than 0",
)
@click.option(
    "-splits",
    "--splits",
    type=Union[List[str], str, None],
    default=None,
    help="The name of the splits to evaluate on. "
    "Can be a string for a single split "
    "or a list of strings for multiple splits.",
)
@click.option(
    "-metrics",
    "--metrics",
    type=Union[List[str], str, None],
    default=None,
    help="The name of the metrics to evaluate on. "
    "Can be a string for a single metric "
    "or a list of strings for multiple metrics.",
)
def main(
    target,
    datasets,
    integration,
    engine_type,
    save_path,
    type_serialization,
    batch_size,
    splits,
    metrics,
):

    _LOGGER.info(f"Target to evaluate: {target}")
    if engine_type:
        _LOGGER.info(f"A pipeline with the engine type: {engine_type} will be created")
    else:
        _LOGGER.info(
            "No engine type specified. The target "
            "will be evaluated using the native framework"
        )

    _LOGGER.info(
        f"Datasets to evaluate on: {datasets}\n"
        f"Batch size: {batch_size}\n"
        f"Splits to evaluate on: {splits}\n"
        f"Metrics to evaluate on: {metrics}"
    )

    result: Result = evaluate(
        target=target,
        datasets=datasets,
        integration=integration,
        engine_type=engine_type,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
    )

    result_formatted = result.formatted

    _LOGGER.info(f"Evaluation done. Results:\n{print_result(result_formatted)}")

    save_path = get_save_path(
        save_path=save_path,
        type_serialization=type_serialization,
        default_file_name="result",
    )
    if save_path:
        _LOGGER.info(f"Saving the evaluation results to {save_path}")
        save_evaluations(
            evaluations=result_formatted,
            save_path=save_path,
            save_format=type_serialization,
        )


def evaluate(
    target: str,
    datasets: Union[str, List[str]],
    integration: str,
    engine_type: Union[
        DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE, None
    ] = DEEPSPARSE_ENGINE,
    batch_size: int = 1,
    splits: Union[List[str], str, None] = None,
    metrics: Union[List[str], str, None] = None,
    **kwargs,
) -> Result:

    eval_integration = EvaluationRegistry.load_from_registry(integration)
    return eval_integration(
        target=target,
        datasets=datasets,
        engine_type=engine_type,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
        **kwargs,
    )


if __name__ == "__main__":
    main()
