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
from typing import Any, List

import numpy
from pydantic import BaseModel, Field

import deepsparse.loggers.metric_functions.built_ins as built_ins
import torch


__all__ = ["FunctionLoggingConfig", "TargetLoggingConfig", "PipelineLoggingConfig"]


def _get_function_callable_and_name(name: str):  # -> Callable[[np.array], np.array]:
    if name.startswith("torch."):
        return getattr(torch, name.split(".")[1]), name.split(".")[1]
    if name.startswith("numpy.") or name.startswith("np."):
        return getattr(numpy, name.split(".")[1]), name.split(".")[1]
    if name.startswith("builtins:"):
        return getattr(built_ins, name.split(":")[1]), name.split(":")[1]
    # assume a dynamic import function of the form '<path>:<name>'
    path, func_name = name.split(":")
    spec = importlib.util.spec_from_file_location("user_defined_metric_functions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name), func_name


class FunctionLoggingConfig(BaseModel):
    func: Any
    function_name: str
    frequency: int

    def __init__(self, **data):
        function_name = data.get("func")
        func, function_name = _get_function_callable_and_name(function_name)
        data["func"] = func
        data["function_name"] = function_name
        super().__init__(**data)


class TargetLoggingConfig(BaseModel):
    target: str
    mappings: List[FunctionLoggingConfig]


class PipelineLoggingConfig(BaseModel):
    name: str
    targets: List[TargetLoggingConfig]
