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
from os import getpid
from typing import Any, Dict, List, Optional, Union

import psutil
from deepsparse.loggers import (
    REQUEST_DETAILS_IDENTIFIER_PREFIX,
    RESOURCE_UTILIZATION_IDENTIFIER_PREFIX,
    BaseLogger,
    MetricCategories,
)
from deepsparse.server.config import SystemLoggingConfig, SystemLoggingGroup
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


_LOGGER = logging.getLogger(__name__)

__all__ = ["log_system_information", "SystemLoggingMiddleware"]


class SystemLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Middleware implementation for logging system metrics.

    A "middleware" is a function that works with every request before
    it is processed by any specific path operation.
    And also with every response before returning it.

    :param app: A FastAPI app instance
    :param server_logger: A server logger instance
    :param system_logging_config: A system logging config instance
    """

    def __init__(
        self,
        app: FastAPI,
        server_logger: BaseLogger,
        system_logging_config: SystemLoggingConfig,
    ):
        super().__init__(app)
        self.server_logger = server_logger
        self.system_logging_config = system_logging_config

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            response = await call_next(request)
        except Exception as err:  # noqa: F841
            log_system_information(
                self.server_logger,
                self.system_logging_config,
                REQUEST_DETAILS_IDENTIFIER_PREFIX,
                response_message=f"{err.__class__.__name__}: {err}",
            )
            log_system_information(
                self.server_logger,
                self.system_logging_config,
                REQUEST_DETAILS_IDENTIFIER_PREFIX,
                successful_request=0,
            )
            _LOGGER.error(err)
            raise

        log_system_information(
            self.server_logger,
            self.system_logging_config,
            REQUEST_DETAILS_IDENTIFIER_PREFIX,
            response_message=f"Response status code: {response.status_code}",
        )
        log_system_information(
            self.server_logger,
            self.system_logging_config,
            REQUEST_DETAILS_IDENTIFIER_PREFIX,
            successful_request=int((response.status_code == 200)),
        )
        return response


def log_resource_utilization(
    server_logger: BaseLogger,
    prefix: str = RESOURCE_UTILIZATION_IDENTIFIER_PREFIX,
    **items_to_log: Dict[str, Any],
):
    """
    Send to the server_logger the logs pertaining to
    the resource utilization of the server process.
    If yes, compute and log the relevant data.

    This includes:
    - CPU utilization
    - Memory utilization
    - Total memory available

    :param server_logger: the logger to log the metrics to
    :param prefix: the prefix to use for the identifier
    :param items_to_log: any additional items to log.
        These will be key-value pairs, where the key is the
        identifier string and the value is the value to log.
    """
    process = psutil.Process(getpid())
    # A float representing the current system-wide CPU utilization as a percentage
    cpu_percent = process.cpu_percent()
    # A float representing process memory utilization as a percentage
    memory_percent = process.memory_percent()
    # Total physical memory
    total_memory_bytes = psutil.virtual_memory().total
    total_memory_megabytes = total_memory_bytes / 1024 / 1024

    identifier_to_value = {
        "cpu_utilization_percent": cpu_percent,
        "memory_utilization_percent": memory_percent,
        "total_memory_available_MB": total_memory_megabytes,
    }
    if items_to_log:
        identifier_to_value.update(items_to_log)

    _send_information_to_logger(
        logger=server_logger, identifier_to_value=identifier_to_value, prefix=prefix
    )


def log_request_details(
    server_logger: BaseLogger,
    prefix: str = REQUEST_DETAILS_IDENTIFIER_PREFIX,
    **items_to_log: Dict[str, Any],
):
    """
    Send to the server_logger the logs pertaining to
    the request_details of the server process.
    Request details information are to be passed as kwargs.
    (where key is the identifier and value is the value to log)

    :param server_logger: the logger to log the metrics to
    :param prefix: the prefix to use for the identifier
    :param items_to_log: The information that is to be logged under this
        particular system logging metric group. The key of `items_to_log` is
        the identifier and value is the value to log.

        For example
        ```
        log_request_details(server_logger,
                            prefix = "request_details"
                            some_identifier = 0.0,
                            some_other_identifier = True)
        ```
        would send:
            value 0.0 under identifier "request_details/some_identifier"
            value True under identifier "request_details/some_other_identifier"
        to the `server_logger`
    """

    _send_information_to_logger(
        logger=server_logger, identifier_to_value=items_to_log, prefix=prefix
    )


# maps the metric group name to the function that logs the information
# pertaining to this metric group name
_PREFIX_MAPPING = {
    REQUEST_DETAILS_IDENTIFIER_PREFIX: log_request_details,
    RESOURCE_UTILIZATION_IDENTIFIER_PREFIX: log_resource_utilization,
}


def log_system_information(
    server_logger: BaseLogger,
    system_logging_config: SystemLoggingConfig,
    system_metric_groups: Optional[Union[str, List[str]]] = None,
    **items_to_log,
):
    """
    A general function that handles logging of
    system information by the server logger

    :param server_logger: the logger to log the metrics to
    :param system_logging_config: a SystemLoggingConfig object that contains
        the configuration for the system logging
    :param system_metric_groups: a name, or a list of names of groups that this function
        should log information for (subset of all the available groups). If None,
        all available groups will be logged (as specified by
        the `system_logging_config`).
    :param items_to_log: any additional items to log. Default is None
        These will be key-value pairs, where the key is the
        identifier string and the value is the value to log.
    """
    if not system_logging_config.enable:
        # system logging disabled; nothing is being logged
        return

    for config_group_name, config_group_args in system_logging_config:
        # iterate over all the valid system logging groups
        if not isinstance(config_group_args, SystemLoggingGroup):
            continue
        if not config_group_args.enable:
            continue
        if system_metric_groups:
            if isinstance(system_metric_groups, str):
                system_metric_groups = [system_metric_groups]
            if config_group_name not in system_metric_groups:
                continue

        # retrieve the function that logs the information for this group
        logging_func = _PREFIX_MAPPING.get(config_group_name)
        if not logging_func:
            continue
        logging_func(server_logger=server_logger, **items_to_log)


def _send_information_to_logger(
    logger: BaseLogger, identifier_to_value: Dict[str, Any], prefix: str
):
    for identifier, value in identifier_to_value.items():
        logger.log(
            identifier=f"{prefix}/{identifier}",
            value=value,
            category=MetricCategories.SYSTEM,
        )
