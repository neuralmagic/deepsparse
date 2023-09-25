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

from pydantic import ValidationError

from deepsparse import Scheduler
from deepsparse.benchmark.config import PipelineBenchmarkConfig


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


class ThreadPinningMode:
    CORE: str = "core"
    NUMA: str = "numa"
    NONE: str = "none"


def decide_thread_pinning(pinning_mode: str) -> None:
    """
    Enable binding threads to cores ('core' the default), threads to cores on sockets
    ('numa'), or disable ('none')"

    :param pinning_mode: thread pinning mode to use
    :return: None
    """
    pinning_mode = pinning_mode.lower()
    if pinning_mode == ThreadPinningMode.CORE:
        os.environ["NM_BIND_THREADS_TO_CORES"] = "1"
        _LOGGER.info("Thread pinning to cores enabled")
    elif pinning_mode == ThreadPinningMode.NUMA:
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "1"
        _LOGGER.info("Thread pinning to socket/numa nodes enabled")
    elif pinning_mode in ThreadPinningMode.NONE:
        os.environ["NM_BIND_THREADS_TO_CORES"] = "0"
        os.environ["NM_BIND_THREADS_TO_SOCKETS"] = "0"
        _LOGGER.info("Thread pinning disabled, performance may be sub-optimal")
    else:
        _LOGGER.info(
            "Recieved invalid option for thread_pinning '%s', skipping" % pinning_mode
        )


def parse_scheduler(scenario: str) -> Scheduler:
    """
    Returns a threading scheduler based on desired scenario

    :param scenario: scheduling scenario to use
    :return: scehduler with desred scenario
    """
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
        _LOGGER.warning(
            "Recieved invalid option for scenario'%s', defaulting to async" % scenario
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
            _LOGGER.warning(
                "num_streams default value chosen of %d. "
                "This requires tuning and may be sub-optimal" % default_num_streams
            )
            return default_num_streams


def parse_input_config(input_config_file: str) -> Dict[str, any]:
    if input_config_file is None:
        _LOGGER.warning("No input configuration file provided, using default.")
        return PipelineBenchmarkConfig()

    config_file = open(input_config_file)
    config = json.load(config_file)
    config_file.close()
    try:
        return PipelineBenchmarkConfig(**config)
    except ValidationError as e:
        _LOGGER.error(e)
        raise e
