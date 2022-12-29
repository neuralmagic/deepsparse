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

from typing import Any

from deepsparse import Pipeline
from deepsparse.loggers import (
    REQUEST_DETAILS_IDENTIFIER_PREFIX,
    BaseLogger,
    MetricCategories,
)
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


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
        except Exception as e:  # noqa: F841
            log_request_details(self.server_logger, successful_request=0)

        log_request_details(
            self.server_logger, successful_request=int((response.status_code == 200))
        )
        return response


def log_resource_utilization(pipeline: Pipeline, **kwargs: Any):
    """
    Scope for 1.4:
    - CPU utilization overall
    - Memory available overall
    - Memory used overall (shall we continuously log this?
      this will be a constant value in time)
    - Number of core used by the pipeline
    """
    pass


def log_request_details(
    server_logger: BaseLogger,
    prefix: str = REQUEST_DETAILS_IDENTIFIER_PREFIX,
    **items_to_log: Any,
):
    """
    Logs the request details of the server process.
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
            value True under idedentifier "request_details/some_other_identifier"
        to the `server_logger`
    """
    for identifier, value in items_to_log.items():
        server_logger.log(
            identifier=f"{prefix}/{identifier}",
            value=value,
            category=MetricCategories.SYSTEM,
        )
