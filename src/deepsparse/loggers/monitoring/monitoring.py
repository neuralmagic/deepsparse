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
import warnings
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy
import numpy as np
from pydantic import BaseModel

from deepsparse.loggers.monitoring import estimators


def extract_logging_data(
    logging_config: Dict[str, Dict[str, Dict[str, Any]]],
    pipeline_inputs: Optional[BaseModel] = None,
    pipeline_outputs: Optional[BaseModel] = None,
    engine_inputs: Optional[Any] = None,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Extracts logging data (estimates) from the pipeline data.

    :param logging_config: Nested dictionary,
        that contains the information
        about the processing that is applied to `pipeline_inputs`,
        `pipeline_outputs` and `engine_inputs`
    :param pipeline_inputs: Inputs to the inference pipeline
    :param pipeline_outputs: Outputs from the inference pipeline
    :param engine_inputs: Direct input to the inference engine
    :return: Nested dictionary, that contains the computed logging data (estimates)
    """
    if pipeline_outputs or engine_inputs:
        raise NotImplementedError

    if logging_config["pipeline_inputs"] and not pipeline_inputs:
        warnings.warn(
            "No `pipeline_inputs` specified, but the `logging_config` "
            "contains non-empty values for 'pipeline_inputs' key."
        )
    pipeline_inputs_numpy = (
        extract_numpy_data(pipeline_inputs) if pipeline_inputs is not None else {}
    )

    pipeline_inputs_estimates = compute_estimates(
        pipeline_inputs_numpy, config=logging_config["pipeline_inputs"]
    )

    engine_inputs_estimates = {}
    pipeline_outputs_estimates = {}

    return {
        "pipeline_inputs": pipeline_inputs_estimates,
        "engine_inputs": engine_inputs_estimates,
        "pipeline_outputs": pipeline_outputs_estimates,
    }


def extract_numpy_data(data_input: Any) -> Dict[str, numpy.ndarray]:
    """
    Extracts the numpy arrays contained within `data_input`

    :param data_input: A data structure that
        contains numpy arrays
    :return: A dictionary where keys are string identifiers
        and the values are the respective numpy arrays
    """
    if isinstance(data_input, BaseModel):
        data_input = dict(data_input)
    else:
        raise NotImplementedError

    data_numpy = {
        identifier: data
        for (identifier, data) in data_input.items()
        if isinstance(data, numpy.ndarray)
    }
    return data_numpy


def compute_estimates(
    input_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Takes `input_data` and applies various estimators (as specified in the config) to it

    :param input_data: A dictionary, where keys are string identifiers
        and values are arbitrary data structures to compute estimates from
    :param config: A dictionary, where keys are the estimator identifiers
        and values are (optional) estimator arguments
    :return: A nested dictionary, with the structure
        {'data_identifier': {'estimator_identifier' : estimate}
    """
    computed_estimates = defaultdict(dict)
    # iterate over estimators specified in the config
    for estimator_identifier, estimator_args in config.items():
        try:
            estimator_func = getattr(estimators, estimator_identifier)
        except Exception as e:
            raise type(e)(
                e.message + f"Unknown estimator identifier: '{estimator_identifier}'. "
                f"The available identifiers are: {estimators.__all__}."
            )

        for data_identifier, data in input_data.items():
            estimate = estimator_func(data, **estimator_args)
            computed_estimates[data_identifier][
                f"{estimator_identifier}_result"
            ] = estimate
    return dict(computed_estimates)
