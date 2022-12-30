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
from typing import Any, Dict

import psutil
from deepsparse.loggers import (
    REQUEST_DETAILS_IDENTIFIER_PREFIX,
    RESOURCE_UTILIZATION_IDENTIFIER_PREFIX,
    BaseLogger,
    MetricCategories,
)
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


_LOGGER = logging.getLogger(__name__)
__all__ = ["log_resource_utilization", "log_request_details", "SystemLoggingMiddleware"]


class SystemLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Middleware implementation for logging system metrics.

    A "middleware" is a function that works with every request before
    it is processed by any specific path operation.
    And also with every response before returning it.

    :param app: A FastAPI app instance
    :param server_logger: A server logger instance
    """

    def __init__(self, app: FastAPI, server_logger: BaseLogger):
        super().__init__(app)
        self.server_logger = server_logger

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            response = await call_next(request)
        except Exception as err:  # noqa: F841
            log_request_details(
                self.server_logger, response_message=f"{err.__class__.__name__}: {err}"
            )
            log_request_details(self.server_logger, successful_request=0)
            _LOGGER.error(err)
            raise

        log_request_details(
            self.server_logger,
            response_message=f"Response status code: {response.status_code}",
        )
        log_request_details(
            self.server_logger, successful_request=int((response.status_code == 200))
        )
        return response


def log_resource_utilization(
    server_logger: BaseLogger,
    prefix: str = RESOURCE_UTILIZATION_IDENTIFIER_PREFIX,
    **items_to_log: Dict[str, Any],
):
    """
    Checks whether server_logger expects to receive logs pertaining to
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
    if not _logging_enabled(server_logger=server_logger, group_name=prefix):
        return
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
    Checks whether server_logger expects to receive logs pertaining to
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
    if not _logging_enabled(server_logger=server_logger, group_name=prefix):
        return

    _send_information_to_logger(
        logger=server_logger, identifier_to_value=items_to_log, prefix=prefix
    )


def _logging_enabled(server_logger: BaseLogger, group_name: str) -> bool:
    function_loggers = server_logger.logger.loggers
    return any(
        [
            logger
            for logger in function_loggers
            if group_name == logger.target_identifier
        ]
    )


def _send_information_to_logger(
    logger: BaseLogger, identifier_to_value: Dict[str, Any], prefix: str
):
    for identifier, value in identifier_to_value.items():
        logger.log(
            identifier=f"{prefix}/{identifier}",
            value=value,
            category=MetricCategories.SYSTEM,
        )
