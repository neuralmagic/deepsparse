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

import json
import logging
import os
from typing import Dict

from deepsparse import Scheduler


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "decide_thread_pinning",
    "parse_scheduler",
    "parse_scenario",
    "parse_num_streams",
    "parse_input_config",
]

DEFAULT_STRING_LENGTH = 50
DEFAULT_IMAGE_SHAPE = (240, 240, 3)


def decide_thread_pinning(pinning_mode: str) -> None:
    pinning_mode = pinning_mode.lower()
    if pinning_mode in "core":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "1"
        _LOGGER.info("Thread pinning to cores enabled")
    elif pinning_mode in "numa":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "1"
        _LOGGER.info("Thread pinning to socket/numa nodes enabled")
    elif pinning_mode in "none":
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "0"
        _LOGGER.info("Thread pinning disabled, performance may be sub-optimal")
    else:
        _LOGGER.info(
            "Recieved invalid option for thread_pinning '{}', skipping".format(
                pinning_mode
            )
        )


def parse_scheduler(scenario: str) -> Scheduler:
    scenario = scenario.lower()
    if scenario == "multistream":
        return Scheduler.multi_stream
    elif scenario == "singlestream":
        return Scheduler.single_stream
    elif scenario == "elastic":
        return Scheduler.elastic
    else:
        return Scheduler.multi_stream


def parse_scenario(scenario: str) -> str:
    scenario = scenario.lower()
    if scenario == "async":
        return "multistream"
    elif scenario == "sync":
        return "singlestream"
    elif scenario == "elastic":
        return "elastic"
    else:
        _LOGGER.info(
            "Recieved invalid option for scenario'{}', defaulting to async".format(
                scenario
            )
        )
        return "multistream"


def parse_num_streams(num_streams: int, num_cores: int, scenario: str):
    # If model.num_streams is set, and the scenario is either "multi_stream" or
    # "elastic", use the value of num_streams given to us by the model, otherwise
    # use a semi-sane default value.
    if scenario == "sync" or scenario == "singlestream":
        if num_streams and num_streams > 1:
            _LOGGER.info("num_streams reduced to 1 for singlestream scenario.")
        return 1
    else:
        if num_streams:
            return num_streams
        else:
            default_num_streams = max(1, int(num_cores / 2))
            _LOGGER.info(
                "num_streams default value chosen of {}. "
                "This requires tuning and may be sub-optimal".format(
                    default_num_streams
                )
            )
            return default_num_streams


def parse_input_config(input_config_file: str) -> Dict[str, any]:
    config_file = open(input_config_file)
    config = json.load(config_file)
    config_file.close()
    return config
