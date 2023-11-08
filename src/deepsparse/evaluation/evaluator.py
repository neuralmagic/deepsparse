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
Module for evaluating models on the various evaluation integrations

##########
Command help:
usage: deepsparse.eval [-h]
                        [-d DATASETS]
                        [-i INTEGRATION]
                        [-b BATCH_SIZE]
                        [-e ENGINE_TYPE]
                        [-s SPLITS]
                        [-m METRICS]
                        target

Evaluate targets on various evaluation integrations

positional arguments:
    target              A path to a remote/local directory containing ONNX/torch model or a SparseZoo stub

optional arguments:
    -h, --help          show this help message and exit
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
                        of a torch model), the engine type should be set to None"
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to use for the evaluation. Must be greater than 0
    -s SPLITS, --splits SPLITS
                        The name of the splits to evaluate on. Can be a string for a single split
                        or a list of strings for multiple splits.
    -m METRICS, --metrics METRICS
                        The name of the metrics to evaluate on. Can be a string for a single metric
                        or a list of strings for multiple metrics.



##########
Example valuation of a Deepsparse pipeline that uses MPT quantized model from SparseZoo.
The evaluation will be run using `lm-evaluation-harness` on `hellaswag` dataset:
deepsparse.eval zoo:mpt-7b-mpt_pretrain-base_quantized \
                --datasets hellaswag \
                --integration lm-evaluation-harness \

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

"""  # noqa: E501

import argparse
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE


__all__ = ["evaluate"]

_LOGGER = logging.getLogger(__name__)


@dataclass
class Metric:
    type: str
    value: float


@dataclass
class Dataset:
    type: str
    name: str
    config: str
    split: str


@dataclass
class EvalSample:
    input: Any
    output: Any


@dataclass
class Evaluation:
    task: str
    dataset: Dataset
    metrics: List[Metric]
    samples: List[EvalSample]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate targets on various evaluation integrations"
    )

    parser.add_argument(
        "target",
        type=str,
        help="A path to a remote or local directory containing ONNX/torch model "
        "(including all the auxiliary files) or a SparseZoo stub",
    )

    parser.add_argument(
        "-d",
        "--datasets",
        type=Union[str, List[str]],
        help="The datasets to evaluate on. Can be a string for a single dataset "
        "or a list of strings for multiple datasets",
    )

    parser.add_argument(
        "-e",
        "--engine_type",
        type=Optional[DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE],
        default=DEEPSPARSE_ENGINE,
        help="The engine to use for the evaluation. The default is the "
        "DeepSparse engine. If the evaluation should be run without "
        "initializing a pipeline (e.g. for the evaluation of a torch "
        "model), the engine type should be set to None",
    )

    parser.add_argument(
        "-s",
        "--splits",
        type=Optional[List[str], str],
        default=None,
        help="The name of the splits to evaluate on. "
        "Can be a string for a single split "
        "or a list of strings for multiple splits.",
    )

    parser.add_argument(
        "-m",
        "--metrics",
        type=Optional[List[str], str],
        default=None,
        help="The name of the metrics to evaluate on. "
        "Can be a string for a single metric "
        "or a list of strings for multiple metrics.",
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
) -> List[Evaluation]:

    _LOGGER.info(f"Target to evaluate: {target}")
    if engine_type:
        _LOGGER.info(
            f"A pipeline with the engine type: " f"{engine_type} will be created"
        )
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

    return eval_integration(
        target=target,
        datasets=datasets,
        engine_type=engine_type,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
        **kwargs,
    )


def main():
    args = parse_args()
    return evaluate(
        target=args.target,
        datasets=args.datasets,
        integration=args.integration,
        engine_type=args.engine_type,
        batch_size=args.batch_size,
        splits=args.splits,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
