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
import re
from typing import Any, Type


def import_from_path(path: str) -> Type[Any]:
    """
    Import the module and the name of the function/class separated by :
    Examples:
      path = "/path/to/file.py:func_name"
      path = "/path/to/file:class_name"
    :param path: path including the file path and object name
    :return Function or class object
    """
    path, class_name = path.split(":")
    _path = path

    path = path.split(".py")[0]
    path = re.sub(r"/+", ".", path)
    try:
        module = importlib.import_module(path)
    except ImportError:
        raise ImportError(f"Cannot find module with path {_path}")

    try:
        return getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Cannot find {class_name} in {_path}")
