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


from os import getpid
from typing import Any, Dict

import psutil
from deepsparse import Pipeline
from deepsparse.loggers import (
    RESOURCE_UTILIZATION_IDENTIFIER_PREFIX,
    BaseLogger,
    MetricCategories,
)


__all__ = ["log_resource_utilization", "log_request_details"]


def log_resource_utilization(
    server_logger: BaseLogger,
    prefix: str = RESOURCE_UTILIZATION_IDENTIFIER_PREFIX,
    **additional_items_to_log: Dict[str, Any],
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
    :param additional_items_to_log: any additional items to log.
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
    if additional_items_to_log:
        identifier_to_value.update(additional_items_to_log)

    _send_information_to_logger(
        logger=server_logger, identifier_to_value=identifier_to_value, prefix=prefix
    )


def log_request_details(pipeline: Pipeline, **kwargs: Any):
    """
    Scope for 1.4:
    - Number of Successful Requests
    (binary events, 1 or 0 per invocation of an endpoint)
    - Batch size
    - Number of Inferences (
    number of successful inferences times the respective batch size)
    """
    pass


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
