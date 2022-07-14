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

import importlib

import pytest


@pytest.fixture()
def yolact_deps():
    """
    Auto delete fixture for yolact dependencies
    """
    yield ("torchvision", "cv2")


def test_autoinstall(yolact_deps):
    import deepsparse.yolact  # noqa F401

    for dependency in yolact_deps:
        _dep = importlib.import_module(dependency)
        assert _dep, (
            f"Expected {dependency} to be autoinstalled, " f"but it was not installed"
        )
