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
from typing import Tuple

import psutil


def log_system_info(
    server_logger: "BaseLogger",
    pipeline: "Pipeline",
):
    batch_size = pipeline.engine.batch_size
    model_path = pipeline.model_path
    num_cores = pipeline.engine.num_cores


def get_resource_utilization() -> Tuple[float, float, int]:
    """
    Returns the percentage that denotes how much CPU is being used
    by the main python thread (server)
    :return:
    """
    process = psutil.Process(getpid())
    # Return a float representing the current system-wide CPU utilization as a percentage
    cpu_percent = process.cpu_percent(interval=None)
    # Compare process memory to total physical system memory and calculate process memory utilization as a percentage
    memory_percent = process.memory_percent()
    # Total physical memory in Bytes
    total_memory_bytes = psutil.virtual_memory().total
    total_memory_megabytes = total_memory_bytes / 1024 / 1024
    return cpu_percent, memory_percent, total_memory_megabytes
