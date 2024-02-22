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

from typing import Any

from deepsparse.middlewares.middleware import MiddlewareCallable


NAME_KEY = "name"


class LoggerMiddleware(MiddlewareCallable):
    def __init__(
        self,
        call_next: MiddlewareCallable,
        identifier: str = "LoggerMiddleware",
    ):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:

        tag = kwargs.get(NAME_KEY)

        inference_state = kwargs.get("inference_state")
        if inference_state and hasattr(inference_state, "logger"):
            logger = inference_state.logger  # metric logger
            rtn = self.call_next(*args, **kwargs)
            logger(
                value=rtn,
                tag=tag,
            )

            return rtn

        return self.call_next(*args, **kwargs)
