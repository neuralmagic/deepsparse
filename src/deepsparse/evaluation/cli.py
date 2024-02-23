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
Usage: deepsparse.eval [OPTIONS] [INTEGRATION_ARGS]...

  Module for evaluating models on the various evaluation integrations

OPTIONS:
    MODEL_PATH
                        A path to an ONNX model, local directory containing ONNX model
                        (including all the auxiliary files) or a SparseZoo stub
    -d DATASETS, --datasets DATASETS
                        The dataset to evaluate on. The user may pass multiple datasets
                        by passing the option multiple times.
    -i INTEGRATION, --integration INTEGRATION
                        Optional name of the evaluation integration to use. Must be a valid
                        integration name that is registered in the evaluation registry
    -e ENGINE_TYPE, --engine_type ENGINE_TYPE
                        Inference engine to use for the evaluation. The default
                        is the DeepSparse engine.
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

INTEGRATION_ARGS:
    Additional, unstructured arguments to pass to the evaluation integration.

#########
EXAMPLES
#########

##########
Example command for evaluating a quantized MPT model from SparseZoo using the Deepsparse Engine.
The evaluation will be run using `lm-evaluation-harness` on `hellaswag` and `gsm8k` datasets:
deepsparse.evaluate zoo:mpt-7b-mpt_pretrain-base_quantized \
                --dataset hellaswag \
                --dataset gsm8k \
                --integration lm-evaluation-harness \
                --limit 2 # limit the number of samples to evaluate on, specific to the integration

"""  # noqa: E501
import logging
from typing import List, Union

import click

from deepsparse.evaluation.evaluator import evaluate
from deepsparse.evaluation.results import Result, save_result
from deepsparse.evaluation.utils import get_save_path, parse_kwarg_tuples
from deepsparse.operators.engine_operator import (
    DEEPSPARSE_ENGINE,
    ORT_ENGINE,
    TORCHSCRIPT_ENGINE,
)


_LOGGER = logging.getLogger(__name__)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.argument(
    "model_path",
    type=click.Path(dir_okay=True, file_okay=True),
    required=True,
)
@click.option(
    "-d",
    "--datasets",
    type=str,
    multiple=True,
    help="The name of dataset to evaluate on. The user may pass multiple "
    "datasets names by passing the option multiple times.",
)
@click.option(
    "-i",
    "--integration",
    type=str,
    required=False,
    help="Optional name of the evaluation integration to use. Must be a valid "
    "integration name that is registered in the evaluation registry",
)
@click.option(
    "-e",
    "--engine_type",
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE]),
    default=DEEPSPARSE_ENGINE,
    help="The engine to use for the evaluation. The default is the "
    "DeepSparse engine. ",
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
@click.argument("integration_args", nargs=-1, type=click.UNPROCESSED)
def main(
    model_path,
    datasets,
    integration,
    engine_type,
    save_path,
    type_serialization,
    batch_size,
    splits,
    metrics,
    integration_args,
):
    """
    Evaluate MODEL_PATH on the various evaluation integrations

    - MODEL_PATH can be path to an ONNX model, local directory
    containing ONNX model (including all the auxiliary files)
    or a SparseZoo stub

    """
    # join datasets to a list if multiple datasets are passed
    datasets = list(datasets) if not isinstance(datasets, str) else datasets
    # format kwargs to a  dict
    integration_args = parse_kwarg_tuples(integration_args)

    _LOGGER.info(
        f"Creating {engine_type} pipeline to evaluate from model path: {model_path}"
    )

    _LOGGER.info(
        f"Datasets to evaluate on: {datasets}\n"
        f"Batch size: {batch_size}\n"
        f"Splits to evaluate on: {splits}\n"
        f"Metrics to evaluate on: {metrics}\n"
        f"Additional integration arguments supplied: {integration_args}"
    )

    result: Result = evaluate(
        model=model_path,
        datasets=datasets,
        integration=integration,
        engine_type=engine_type,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
        **integration_args,
    )

    _LOGGER.info(f"Evaluation done. Results:\n{result.formatted}")

    save_path = get_save_path(
        save_path=save_path,
        type_serialization=type_serialization,
        file_name="result",
    )
    if save_path:
        _LOGGER.info(f"Saving the evaluation results to {save_path}")
        save_result(result=result, save_path=save_path, save_format=type_serialization)


if __name__ == "__main__":
    main()
