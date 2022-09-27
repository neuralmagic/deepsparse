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


import multiprocessing
import time
from pathlib import Path

import requests
import yaml

import pytest
from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.server import start_server

from .test_prometheus import _find_free_port


@pytest.fixture(scope="function")
def server_port():
    p = _find_free_port()
    yield p


@pytest.fixture(scope="function")
def config_path(tmp_path: Path):
    cfg = ServerConfig(endpoints=[], num_cores=1, num_workers=1, loggers=None)
    cfg_path = str(tmp_path / "cfg.yaml")
    with open(cfg_path, "w") as fp:
        yaml.safe_dump(cfg.dict(), fp)
    yield cfg_path


@pytest.fixture(scope="function")
def server_process(config_path, server_port):
    proc = multiprocessing.Process(
        target=start_server,
        kwargs=dict(
            config_path=config_path,
            host="0.0.0.0",
            port=server_port,
            hot_reload_config=True,
        ),
    )
    proc.start()
    yield proc
    proc.kill()
    proc.join()


def test_hot_reload_config(server_process, server_port, config_path):
    # wait for server to spin up
    time.sleep(2.0)

    resp = requests.get(f"http://localhost:{server_port}/")
    assert resp.status_code == 200

    # endpoint doesn't exist yet
    resp = requests.post(f"http://localhost:{server_port}/predict1")
    assert resp.status_code == 404

    cfg = ServerConfig(endpoints=[], num_cores=1, num_workers=1, loggers=None)
    cfg.endpoints.append(
        EndpointConfig(
            route="/predict1",
            task="qa",
            model="default",
            kwargs=dict(engine_type="onnxruntime"),
        )
    )
    with open(config_path, "w") as fp:
        yaml.safe_dump(cfg.dict(), fp)

    # wait for endpoint to be added to server
    time.sleep(2.0)

    # the endpoint should exist now
    resp = requests.post(
        f"http://localhost:{server_port}/predict1",
        json={"question": "who am i", "context": "i am bob"},
    )
    assert resp.status_code == 200

    cfg.endpoints.clear()
    with open(config_path, "w") as fp:
        yaml.safe_dump(cfg.dict(), fp)

    # wait for endpoint to be removed from the server
    time.sleep(1.0)

    # the endpoint should not exist anymore
    resp = requests.post(f"http://localhost:{server_port}/predict1")
    assert resp.status_code == 404
