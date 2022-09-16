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

from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.server import _build_app
from fastapi.testclient import TestClient
from tests.utils import mock_engine


def _find_free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 0))
    portnum = s.getsockname()[1]
    s.close()

    return portnum


@mock_engine(rng_seed=0)
def test_instantiate_prometheus(tmp_path):
    client = TestClient(
        _build_app(
            ServerConfig(
                endpoints=[EndpointConfig(task="text_classification", model="default")],
                loggers=dict(
                    prometheus={
                        "port": _find_free_port(),
                        "text_log_save_dir": str(tmp_path),
                        "text_log_save_freq": 30,
                    }
                ),
            )
        )
    )
    r = client.post("/predict", json=dict(sequences="asdf"))
    assert r.status_code == 200
