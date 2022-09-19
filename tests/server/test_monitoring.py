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

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepsparse.server.config import EndpointConfig, ServerConfig, ImageSizesConfig
from deepsparse.server.monitoring import _ContentMonitor, _update_endpoints


def test_no_route_not_in_diff():
    no_route = EndpointConfig(task="b", model="c")
    old = ServerConfig(endpoints=[])
    new = ServerConfig(endpoints=[no_route])

    added, removed = old.endpoint_diff(new)
    assert added == []
    assert removed == []

    added, removed = new.endpoint_diff(old)
    assert added == []
    assert removed == []


def test_added_removed_endpoint_diff():
    route1 = EndpointConfig(task="b", model="c", route="1")
    route2 = EndpointConfig(task="b", model="c", route="2")
    route3 = EndpointConfig(task="b", model="c", route="3")
    old = ServerConfig(endpoints=[route1, route2])
    new = ServerConfig(endpoints=[route1, route3])

    added, removed = old.endpoint_diff(new)
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
        added, removed = old.endpoint_diff(new)
        assert added == [route2]
        assert removed == [route1]


@patch("requests.post")
@patch("requests.delete")
def test_update_endpoints(delete: MagicMock, post: MagicMock):
    route1 = EndpointConfig(task="b", model="c", route="1")
    route2 = EndpointConfig(task="b", model="c", route="2")
    route3 = EndpointConfig(task="b", model="c", route="3")
    old = ServerConfig(num_cores=1, num_workers=1, endpoints=[route1, route2])
    new = ServerConfig(num_cores=1, num_workers=1, endpoints=[route1, route3])

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
