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
from typing import Any

import psutil
from deepsparse import Pipeline
from deepsparse.loggers import BaseLogger, MetricCategories


__all__ = ["log_resource_utilization", "log_request_details"]


def log_resource_utilization(server_logger: BaseLogger):
    """
    Logs the resource utilization of the server process.
    This includes:
    - CPU utilization
    - Memory utilization
    - Total memory available

    :param server_logger: the logger to log the metrics to
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
        "cpu_utilization_[%]": cpu_percent,
        "memory_utilization_[%]": memory_percent,
        "total_memory_available_[MB]": total_memory_megabytes,
    }

    for identifier, value in identifier_to_value.items():
        server_logger.log(
            identifier=identifier, value=value, category=MetricCategories.SYSTEM
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
