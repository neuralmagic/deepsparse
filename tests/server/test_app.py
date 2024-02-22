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

import os
from copy import deepcopy
from re import escape
from unittest.mock import patch

import pytest
from deepsparse.server.cli import _fetch_server
from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.server import Server


def test_add_multiple_endpoints_with_same_route():
    with pytest.raises(ValueError, match="asdf specified 2 times"):
        server = Server(
            ServerConfig(
                num_cores=1,
                num_workers=1,
                endpoints=[
                    EndpointConfig(task="", model="", route="asdf"),
                    EndpointConfig(task="", model="", route="asdf"),
                ],
                loggers={},
            )
        )
        server._build_app()


def test_invalid_integration():
    with pytest.raises(
        ValueError,
        match=escape(
            "asdf is not a supported integration. Must be "
            "one of ['local', 'sagemaker', 'openai']"
        ),
    ):
        server = ServerConfig(
            num_cores=1,
            num_workers=1,
            integration="asdf",
            endpoints=[],
            loggers={},
        )
        _fetch_server("local", server)


def test_pytorch_num_threads():
    torch = pytest.importorskip("torch")

    orig_num_threads = torch.get_num_threads()
    server = Server(
        ServerConfig(
            num_cores=1,
            num_workers=1,
            pytorch_num_threads=None,
            endpoints=[],
            loggers={},
        )
    )
    server._build_app()
    assert torch.get_num_threads() == orig_num_threads

    server = Server(
        ServerConfig(
            num_cores=1,
            num_workers=1,
            pytorch_num_threads=1,
            endpoints=[],
            loggers={},
        )
    )
    server._build_app()
    assert torch.get_num_threads() == 1


@patch.dict(os.environ, deepcopy(os.environ))
def test_thread_pinning_none():
    os.environ.pop("NM_BIND_THREADS_TO_CORES", None)
    os.environ.pop("NM_BIND_THREADS_TO_SOCKETS", None)
    server = Server(
        ServerConfig(
            num_cores=1,
            num_workers=1,
            engine_thread_pinning="none",
            endpoints=[],
            loggers={},
        )
    )
    server._build_app()
    assert os.environ["NM_BIND_THREADS_TO_CORES"] == "0"
    assert os.environ["NM_BIND_THREADS_TO_SOCKETS"] == "0"


@patch.dict(os.environ, deepcopy(os.environ))
def test_thread_pinning_numa():
    os.environ.pop("NM_BIND_THREADS_TO_CORES", None)
    os.environ.pop("NM_BIND_THREADS_TO_SOCKETS", None)
    server = Server(
        ServerConfig(
            num_cores=1,
            num_workers=1,
            engine_thread_pinning="numa",
            endpoints=[],
            loggers={},
        )
    )
    server._build_app()
    assert os.environ["NM_BIND_THREADS_TO_CORES"] == "0"
    assert os.environ["NM_BIND_THREADS_TO_SOCKETS"] == "1"


@patch.dict(os.environ, deepcopy(os.environ))
def test_thread_pinning_cores():
    os.environ.pop("NM_BIND_THREADS_TO_CORES", None)
    os.environ.pop("NM_BIND_THREADS_TO_SOCKETS", None)
    server = Server(
        ServerConfig(
            num_cores=1,
            num_workers=1,
            engine_thread_pinning="core",
            endpoints=[],
            loggers={},
        )
    )
    server._build_app()
    assert os.environ["NM_BIND_THREADS_TO_CORES"] == "1"
    assert os.environ["NM_BIND_THREADS_TO_SOCKETS"] == "0"


def test_invalid_thread_pinning():
    with pytest.raises(ValueError, match='Expected one of {"core","numa","none"}.'):
        server = Server(
            ServerConfig(
                num_cores=1,
                num_workers=1,
                engine_thread_pinning="asdf",
                endpoints=[],
                loggers={},
            )
        )
        server._build_app()
