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


def import_from_registry(name: str):
    registry = "src.deepsparse.loggers_v2.registry.__init__"
    module = importlib.import_module(registry)
    try:
        return getattr(module, name)
    except Exception:
        raise ValueError(f"Cannot import class/func with name '{name}' from {registry}")


def import_from_path(path: str):
    path, class_name = path.split(":")
    path = path.split(".py")[0]

    _path = path
    path = path.replace(r"/", ".")
    try:
        module = importlib.import_module(path)
    except Exception:
        raise ValueError(f"Cannot find module with path {_path}")

    try:
        return getattr(module, class_name)
    except Exception:
        raise ValueError(f"Cannot find {class_name} in {_path}")
