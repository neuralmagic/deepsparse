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

from http import HTTPStatus
from typing import Dict, List, Optional, Union

import numpy
from pydantic import BaseModel

from deepsparse import (
    BaseLogger,
    build_logger,
    get_target_identifier,
    system_logging_config_to_groups,
)
from deepsparse.loggers.config import MetricFunctionConfig, SystemLoggingGroup
from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.protocol import ErrorResponse
from fastapi.responses import JSONResponse


__all__ = [
    "create_error_response",
    "server_logger_from_config",
    "prep_outputs_for_serialization",
]


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )


def server_logger_from_config(config: ServerConfig) -> BaseLogger:
    """
    Builds a DeepSparse Server logger from the ServerConfig.

    :param config: the Server configuration model.
        This configuration by default contains three fields relevant
        for the instantiation of a Server logger:
        - ServerConfig.loggers: this is a configuration of the
            "leaf" loggers (that log information to the end destination)
            that will be used by the Server logger
        - ServerConfig.data_logging: this is a configuration of
            the function loggers, responsible for system logging
            functionality. If present, those function logger wrap
            around the appropriate "leaf" loggers.
        - ServerConfig.endpoints: this is a configuration of the
            endpoints that the Server will serve. Each endpoint
            can have its own data_logging configuration, which
            will be merged parsed out by the logic of this function
    :return: a DeepSparse logger instance
    """

    system_logging_groups = _extract_system_logging_from_endpoints(config.endpoints)
    system_logging_groups.update(system_logging_config_to_groups(config.system_logging))

    return build_logger(
        system_logging_config=system_logging_groups,
        loggers_config=config.loggers,
        data_logging_config=_extract_data_logging_from_endpoints(config.endpoints),
    )


def prep_outputs_for_serialization(
    pipeline_outputs: Union[BaseModel, numpy.ndarray, list]
) -> Union[BaseModel, list]:
    """
    Prepares a pipeline output for JSON serialization by converting any numpy array
    field to a list. For large numpy arrays, this operation will take a while to run.

    :param pipeline_outputs: output data to that is to be processed before
        serialisation. Nested objects are supported.
    :return: Pipeline_outputs with potential numpy arrays
        converted to lists
    """
    if isinstance(pipeline_outputs, BaseModel):
        for field_name in pipeline_outputs.__fields__.keys():
            field_value = getattr(pipeline_outputs, field_name)
            if isinstance(field_value, (numpy.ndarray, BaseModel, list)):
                setattr(
                    pipeline_outputs,
                    field_name,
                    prep_outputs_for_serialization(field_value),
                )

    elif isinstance(pipeline_outputs, numpy.ndarray):
        pipeline_outputs = pipeline_outputs.tolist()

    elif isinstance(pipeline_outputs, list):
        for i, value in enumerate(pipeline_outputs):
            pipeline_outputs[i] = prep_outputs_for_serialization(value)

    return pipeline_outputs


def _extract_system_logging_from_endpoints(
    endpoints: List[EndpointConfig],
) -> Dict[str, SystemLoggingGroup]:
    system_logging_groups = {}
    for endpoint in endpoints:
        if endpoint.logging_config is None:
            continue
        system_logging_groups.update(
            system_logging_config_to_groups(
                endpoint.logging_config, endpoint_name=endpoint.name
            )
        )
    return system_logging_groups


def _extract_data_logging_from_endpoints(
    endpoints: List[EndpointConfig],
) -> Optional[Dict[str, List[MetricFunctionConfig]]]:
    data_logging = {}
    for endpoint in endpoints:
        if endpoint.data_logging is None:
            continue
        for target, metric_functions in endpoint.data_logging.items():
            # if needed, get the new target identifier from
            # the target and the endpoint name
            new_target = get_target_identifier(
                target_name=target, pipeline_identifier=endpoint.name
            )
            data_logging[new_target] = metric_functions
    # if no data logging data specified, return None
    return data_logging if data_logging else None
