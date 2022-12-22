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
from deepsparse.loggers import BaseLogger, MetricCategories
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


__all__ = ["log_resource_utilization", "log_request_details", "SystemLoggingMiddleware"]

REQUEST_DETAILS_IDENTIFIER_PREFIX = "request_details"


class SystemLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, server_logger):
        super().__init__(app)
        self.server_logger = server_logger

    async def dispatch(self, request: Request, call_next):
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
    **kwargs: Any,
):
    """
    Logs the request details of the server process.
    Request details information are to be passed as kwargs.
    (where key is the identifier and value is the value to log)

    :param server_logger: the logger to log the metrics to
    """
    for identifier, value in kwargs.items():
        server_logger.log(
            identifier="/".join([prefix, identifier]),
            value=value,
            category=MetricCategories.SYSTEM,
        )
