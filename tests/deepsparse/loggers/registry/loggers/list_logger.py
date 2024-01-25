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


from typing import Any, Callable


class ListLogger:
    def __init__(self, **_ignore):
        self.logs = []

    def log(
        self,
        value: Any,
        tag: str,
        func: Callable,
        log_type: str,
        **kwargs,
    ):
        placeholders = f"[{log_type}.{tag}.{str(func)}]"
        if (run_time := kwargs.get("run_time")) is not None:
            placeholders += f"[⏱️{run_time}]"
        if (capture := kwargs.get("capture")) is not None:
            placeholders += f" {func}({capture})"

        self.logs.append(f"{placeholders}: {value}")
