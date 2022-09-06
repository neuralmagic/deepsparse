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

import yaml

from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.monitoring import _ContentMonitor, _update_endpoints


@patch("requests.post")
@patch("requests.delete")
def test_update_endpoints(delete: MagicMock, post: MagicMock):
    no_route = EndpointConfig(name="a", task="b", model="c")
    route1 = EndpointConfig(name="a", task="b", model="c", route="1")
    route2 = EndpointConfig(name="a", task="b", model="c", route="2")
    route3 = EndpointConfig(name="a", task="b", model="c", route="3")
    old = ServerConfig(num_cores=1, num_workers=1, endpoints=[no_route, route1, route2])
    new = ServerConfig(num_cores=1, num_workers=1, endpoints=[route1, route3])

    old_s = yaml.dump(old.dict())
    new_s = yaml.dump(new.dict())

    # NOTE: no_route not included in removed since we can't detect
    # changes for this without route specified
    added, removed = _update_endpoints(url="", diff=(old_s, new_s))
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

    assert content.diff() is None

    time.sleep(0.1)
    path.write_text("first")
    assert content.diff() == ("", "first")

    time.sleep(0.1)
    assert content.diff() is None

    time.sleep(0.1)
    path.write_text("second")
    assert content.diff() == ("first", "second")
