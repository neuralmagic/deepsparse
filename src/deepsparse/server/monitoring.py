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
import multiprocessing
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pydantic
import requests
import yaml

from deepsparse.server.config import EndpointConfig, ServerConfig


_LOGGER = logging.getLogger(__name__)


def start_file_watcher(
    config_path: str, endpoints_url: str, check_interval_s: float
) -> multiprocessing.Process:
    """
    Creates and starts a separate process that monitors `config_path`
    for changes every `check_interval_s` seconds.
    If endpoints are modified it sends requests to `endpoints_url`
    to update the server.

    :param config_path: Path to file to watch
    :param endpoints_url: Url to send requests to for updating
    :param check_interval_s: Seconds between each check.
    :return: A **started** daemon `Process`.
    """

    proc = multiprocessing.Process(
        target=_file_watcher,
        kwargs=dict(
            config_path=config_path,
            endpoints_url=endpoints_url,
            check_interval_s=check_interval_s,
        ),
        daemon=True,
    )
    proc.start()
    return proc


def _file_watcher(config_path: str, endpoints_url: str, check_interval_s: float):
    content = _ContentMonitor(config_path)

    while True:
        time.sleep(check_interval_s)
        diff = content.diff()
        if diff is None:
            continue

        try:
            _update_endpoints(endpoints_url, diff)
        except yaml.error.YAMLError:
            _LOGGER.error("Failed to read yaml, not updating.", exc_info=1)
        except pydantic.ValidationError:
            _LOGGER.error("Unable to load ServerConfig, not updating.", exc_info=1)
        except requests.RequestException:
            _LOGGER.error("Requests to server failed, not updating.", exc_info=1)


def _update_endpoints(url: str, diff: Tuple[str, str]) -> None:
    old_content, new_content = diff

    old_config = ServerConfig(**yaml.safe_load(old_content))
    new_config = ServerConfig(**yaml.safe_load(new_content))

    added, removed = _endpoint_diff(old_config, new_config)

    for endpoint in removed:
        _LOGGER.info(f"Requesting removal of endpoint '{endpoint.route}'")
        requests.delete(url, json=endpoint.dict()).raise_for_status()

    for endpoint in added:
        _LOGGER.info(f"Requesting addition of endpoint '{endpoint.route}'")
        requests.post(url, json=endpoint.dict()).raise_for_status()

    return added, removed


class _ContentMonitor:
    """
    Checks `path` for modifications for every call

    :param path: The path to check.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.last_modified = self.path.stat().st_mtime_ns
        self.content = self.path.read_text()

    def diff(self) -> Optional[Tuple[str, str]]:
        mtime = self.path.stat().st_mtime_ns
        if mtime != self.last_modified:
            _LOGGER.info(f"Detected change in {self.path}")

            old_content = self.content
            new_content = self.path.read_text()
            self.content = new_content
            self.last_modified = mtime

            return old_content, new_content


def _endpoint_diff(
    old: ServerConfig, new: ServerConfig
) -> Tuple[List[EndpointConfig], List[EndpointConfig]]:
    old_routes = set([e.route for e in old.endpoints if e.route is not None])
    new_routes = set([e.route for e in new.endpoints if e.route is not None])
    added_routes = new_routes - old_routes
    removed_routes = old_routes - new_routes
    added_endpoints = [e for e in new.endpoints if e.route in added_routes]
    removed_endpoints = [e for e in old.endpoints if e.route in removed_routes]
    return added_endpoints, removed_endpoints
