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
from unittest.mock import MagicMock, patch

import requests
import yaml

import pytest
from deepsparse.server.config import EndpointConfig, ImageSizesConfig, ServerConfig
from deepsparse.server.config_hot_reloading import (
    _ContentMonitor,
    _diff_generator,
    _update_endpoints,
    endpoint_diff,
)
from deepsparse.server.server import start_server
from tests.helpers import wait_for_server

from .test_prometheus import _find_free_port


def test_no_route_not_in_diff():
    no_route = EndpointConfig(task="b", model="c")
    old = ServerConfig(endpoints=[])
    new = ServerConfig(endpoints=[no_route])

    added, removed = endpoint_diff(old, new)
    assert added == []
    assert removed == []

    added, removed = endpoint_diff(new, old)
    assert added == []
    assert removed == []


def test_added_removed_endpoint_diff():
    route1 = EndpointConfig(task="b", model="c", route="1")
    route2 = EndpointConfig(task="b", model="c", route="2")
    route3 = EndpointConfig(task="b", model="c", route="3")
    old = ServerConfig(endpoints=[route1, route2])
    new = ServerConfig(endpoints=[route1, route3])

    added, removed = endpoint_diff(old, new)
    assert added == [route3]
    assert removed == [route2]


def test_endpoint_diff_modified_model():
    default_cfg = dict(model="a", route="1", task="b")
    route1 = EndpointConfig(**default_cfg)
    old = ServerConfig(endpoints=[route1])

    all_fields = dict(
        model="b",
        task="c",
        batch_size=2,
        bucketing=ImageSizesConfig(image_sizes=[], kwargs=dict(a=2)),
    )
    for key, value in all_fields.items():
        cfg = default_cfg.copy()
        cfg[key] = value
        route2 = EndpointConfig(**cfg)
        new = ServerConfig(endpoints=[route2])
        added, removed = endpoint_diff(old, new)
        assert added == [route2]
        assert removed == [route1]


@patch("requests.post")
@patch("requests.delete")
def test_update_endpoints(delete: MagicMock, post: MagicMock):
    route1 = EndpointConfig(task="b", model="c", route="1")
    route2 = EndpointConfig(task="b", model="c", route="2")
    route3 = EndpointConfig(task="b", model="c", route="3")
    old = ServerConfig(endpoints=[route1, route2])
    new = ServerConfig(endpoints=[route1, route3])

    # NOTE: no_route not included in removed since we can't detect
    # changes for this without route specified
    added, removed = _update_endpoints("", old, new)
    assert added == [route3]
    assert removed == [route2]

    delete.assert_called_once_with("", json=route2.dict())
    post.assert_called_once_with("", json=route3.dict())


def test_file_changes(tmp_path: Path):
    # NOTE: this sleeps between each write because timestamps
    # only have a certain resolution

    path = tmp_path / "file.txt"
    path.write_text("")

    content = _ContentMonitor(path)

    assert content.maybe_update_content() is None

    time.sleep(0.1)
    path.write_text("first")
    assert content.maybe_update_content() == ("", "first")

    time.sleep(0.1)
    assert content.maybe_update_content() is None

    time.sleep(0.1)
    path.write_text("second")
    assert content.maybe_update_content() == ("first", "second")


@patch("requests.post")
@patch("requests.delete")
def test_file_monitoring(delete_mock, post_mock, tmp_path: Path):
    path = str(tmp_path / "cfg.yaml")
    versions_path = tmp_path / "cfg.yaml.versions"

    cfg1 = ServerConfig(endpoints=[])
    with open(path, "w") as fp:
        yaml.safe_dump(cfg1.dict(), fp)

    diffs = _diff_generator(path, "", 0.1)
    assert next(diffs) is None
    assert not versions_path.exists()

    cfg2 = ServerConfig(endpoints=[EndpointConfig(task="a", model="b", route="1")])
    with open(path, "w") as fp:
        yaml.safe_dump(cfg2.dict(), fp)
    assert next(diffs) == (cfg1, cfg2, path + ".versions/0.yaml")
    assert versions_path.exists()

    assert next(diffs) is None

    cfg3 = ServerConfig(endpoints=[EndpointConfig(task="a", model="c", route="1")])
    with open(path, "w") as fp:
        yaml.safe_dump(cfg3.dict(), fp)
    assert next(diffs) == (cfg2, cfg3, path + ".versions/1.yaml")

    all_files = sorted(map(str, tmp_path.rglob("*")))
    all_files = [f.replace(str(tmp_path), "") for f in all_files]
    assert all_files == [
        "/cfg.yaml",
        "/cfg.yaml.versions",
        "/cfg.yaml.versions/0.yaml",
        "/cfg.yaml.versions/1.yaml",
    ]

    for idx, v in enumerate(["0.yaml", "1.yaml"]):
        with open(str(versions_path / v)) as fp:
            content = fp.read()
            assert content.startswith(f"# Version {idx} saved at")
            yaml.safe_load(content)


@pytest.fixture
def server_port():
    p = _find_free_port()
    yield p


@pytest.fixture
def config_path(tmp_path: Path):
    cfg = ServerConfig(endpoints=[], num_cores=1, num_workers=1, loggers=None)
    cfg_path = str(tmp_path / "cfg.yaml")
    with open(cfg_path, "w") as fp:
        yaml.safe_dump(cfg.dict(), fp)
    yield cfg_path


@pytest.fixture
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


def test_hot_reload_config_with_start_server(server_process, server_port, config_path):
    assert wait_for_server(f"http://localhost:{server_port}", retries=50, interval=0.1)

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
    for _ in range(50):
        time.sleep(0.1)
        resp = requests.post(f"http://localhost:{server_port}/predict1")
        if resp.status_code != 404:
            break

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
