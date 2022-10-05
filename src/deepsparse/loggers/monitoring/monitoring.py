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

import logging
from collections import defaultdict
from typing import Any, Dict, Set, Union

import numpy
from pydantic import BaseModel

from . import estimators


SUPPORTED_TYPES_COMPUTE_ESTIMATES = {numpy.ndarray, bool, int, float}


def convert_data_to_estimates(
    input_data: Union[BaseModel, Any], config: Dict[str, Any]
) -> Dict[str, Dict[str, numpy.array]]:
    """
    Converts any incoming data from the pipeline into low-dimensional estimates that can be
    further processed by server loggers

    :param input_data: Any type of data returned by running the pipeline with "monitoring" feature
    :param config: A dictionary that specifies the estimates to be computed, along with additional args
    :return: A dictionary that relates the `input_data` with its computed estimates
    """
    monitoring_estimates = defaultdict()
    numpy_inputs = convert_data_to_numpy(
        input_data=input_data, supported_types=SUPPORTED_TYPES_COMPUTE_ESTIMATES
    )

    for input_name, numpy_array in numpy_inputs.items():
        # right now the config is being applied to all the data in input_data
        monitoring_estimates[input_name] = compute_estimates(
            input=numpy_array, config=config
        )
    return monitoring_estimates


def convert_data_to_numpy(
    input_data: Any, supported_types: Set
) -> Dict[str, numpy.ndarray]:
    """
    Converts any `input_data` in a dictionary {data_name: numpy_array}.

    :param input_data: Any type of data returned by running the pipeline with "monitoring" feature
    :param supported_types: Possible types of input_data that will be converted into numpy arrays.
        If input_data is not included in the set of `supported_types`, it will be discarded (with a warning)
    :return: A dictionary that relates the data name with it's numpy array
    """
    if isinstance(input_data, BaseModel):
        # convert Pydantic model to a dictionary
        input_data = dict(input_data)
    else:
        raise NotImplementedError

    for input_name, any_inputs in input_data.items():
        if isinstance(any_inputs, list):
            if len(any_inputs) != 1:
                raise NotImplementedError
            else:
                any_inputs = any_inputs[0]
        if type(any_inputs) not in supported_types:
            logging.warning("boo")
        else:
            input_data[input_name] = (
                numpy.array(any_inputs)
                if not isinstance(any_inputs, numpy.ndarray)
                else any_inputs
            )
    return input_data


def compute_estimates(input: numpy.ndarray, config: Dict[str, Any]):
    results = {}

    for estimator_name, estimator_args in config.items():
        try:
            estimator_func = getattr(estimators, estimator_name)
        except:
            raise ValueError
        results[f"{estimator_name}_result"] = estimator_func(input, **estimator_args)

    return results
