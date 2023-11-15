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
    -m METRICS, --metrics METRICS
                        The name of the metrics to evaluate on. Can be a string for a single metric
                        or a list of strings for multiple metrics.
    -splits SPLITS, --splits SPLITS
                        The name of the splits to evaluate on. Can be a string for a single split
                        or a list of strings for multiple splits.
    --enforce_result_structure --enforce-result-structure
                        Specifies whether to unify all the
                        results into the predefined Evaluation structure. If True, the
                        results will be returned as a list of Evaluation objects.
                        Otherwise, the result will preserve the original result structure
                        from the evaluation integration.
    -h, --help          Show this help message and exit

#########
EXAMPLES
#########

##########
Example command for evaluating a quantized MPT model from SparseZoo using the Deepsparse Engine.
The evaluation will be run using `lm-evaluation-harness` on `hellaswag` dataset:
deepsparse.eval zoo:mpt-7b-mpt_pretrain-base_quantized \
                --datasets hellaswag \
                --integration lm-evaluation-harness \

"""  # noqa: E501
import logging
from typing import Any, List, Optional, Union

import click

from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.evaluation.results import (
    Evaluation,
    print_result,
    save_evaluation,
    validate_result_structure,
)
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
    "be saved under the name 'result.yaml`/'result.json' depending on the serialization type. If argument is not provided, the results will be saved in the current directory",
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
    "-splits",
    "--splits",
    type=Union[List[str], str, None],
    default=None,
    help="The name of the splits to evaluate on. "
    "Can be a string for a single split "
    "or a list of strings for multiple splits.",
)
@click.option(
    "-m",
    "--metrics",
    type=Union[List[str], str, None],
    default=None,
    help="The name of the metrics to evaluate on. "
    "Can be a string for a single metric "
    "or a list of strings for multiple metrics.",
)
@click.option(
    "--enforce-result-structure",
    "--enforce_result_structure",
    default=True,
    help="Specifies whether to unify all the results "
    "into the predefined Evaluation structure. "
    "If True, the results will be returned as a "
    "list of Evaluation objects. Otherwise, the "
    "result will preserve the original result "
    "structure from the evaluation integration.",
)
def main(
    target: str,
    datasets: Union[str, List[str]],
    integration: str,
    engine_type: Union[
        DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE, None
    ] = DEEPSPARSE_ENGINE,
    save_path: str = "result",
    type_serialization: str = "json",
    batch_size: int = 1,
    splits: Union[List[str], str, None] = None,
    metrics: Union[List[str], str, None] = None,
    enforce_result_structure: bool = True,
    **kwargs,
) -> Union[List[Evaluation], Any]:

    _LOGGER.info(f"Target to evaluate: {target}")
    if engine_type:
        _LOGGER.info(f"A pipeline with the engine type: {engine_type} will be created")
    else:
        _LOGGER.info(
            "No engine type specified. The target "
            "will be evaluated using the default framework"
        )
    _LOGGER.info(f"Datasets to evaluate on: {datasets}")
    _LOGGER.info(
        f"Batch size: {batch_size}\n"
        f"Splits to evaluate on: {splits}\n"
        f"Metrics to evaluate on: {metrics}"
    )

    eval_integration = EvaluationRegistry.load_from_registry(integration)

    _LOGGER.info(
        f"The following evaluation integration has "
        f"been successfully setup: {eval_integration.__name__}"
    )

    result = eval_integration(
        target=target,
        datasets=datasets,
        engine_type=engine_type,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
        original_result_structure=enforce_result_structure,
        **kwargs,
    )

    if not enforce_result_structure:
        _LOGGER.info(f"Evaluation done. Results:\n{result}")
        return result

    if not validate_result_structure(result):
        raise ValueError(
            "The evaluation integration must return a list of Evaluation objects "
            "when enforce_result_structure is True."
        )

    _LOGGER.info(f"Evaluation done. Results:\n{print_result(result)}")
    save_path = get_save_path(
        save_path=save_path,
        type_serialization=type_serialization,
        default_file_name="result",
    )
    if save_path:
        _LOGGER.info(f"Saving the evaluation results to {save_path}")
        save_evaluation(
            evaluations=result, save_path=save_path, save_format=type_serialization
        )

    return result


if __name__ == "__main__":
    main()
