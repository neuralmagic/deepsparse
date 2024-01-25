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


import pytest
from deepsparse.loggers.utils import (
    LOGGER_REGISTRY,
    import_from_path,
    import_from_registry,
)


@pytest.mark.parametrize(
    "name, is_successful",
    [
        ("PythonLogger", True),
        ("max", True),
        ("blah", False),
    ],
)
def test_import_from_registry(name, is_successful):
    if is_successful:
        assert import_from_registry(name) is not None
    else:
        with pytest.raises(AttributeError):
            import_from_registry(name)


@pytest.mark.parametrize(
    "path, is_successful",
    [
        (f"{LOGGER_REGISTRY}.py:PythonLogger", True),
        (f"{LOGGER_REGISTRY}:PythonLogger", True),
        ("foo/bar:blah", False),
        (f"{LOGGER_REGISTRY}:blah", False),
    ],
)
def test_import_from_path(path, is_successful):
    if is_successful:
        assert import_from_path(path) is not None
    else:
        with pytest.raises((AttributeError, ImportError)):
            import_from_path(path)
