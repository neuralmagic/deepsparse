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

from typing import Dict, List, Optional

from deepsparse import BaseLogger, build_logger, get_target_identifier
from deepsparse.loggers.config import MetricFunctionConfig
from deepsparse.server.config import EndpointConfig, ServerConfig


__all__ = ["server_logger_from_config"]


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

    return build_logger(
        system_logging_config=config.system_logging,
        loggers_config=config.loggers,
        data_logging_config=_extract_data_logging_from_endpoints(config.endpoints),
    )


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
