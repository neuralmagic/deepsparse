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

import threading
from typing import Any, Dict, Iterator, Optional, Protocol, Sequence


class MiddlewareCallable(Protocol):
    def __call__(self, *args, **kwargs):
        ...

    # @abstractmethod
    def send(self, dct: Dict):
        """Update middleware Manager state"""
        ...


class MiddlewareSpec:
    def __init__(self, cls: type[MiddlewareCallable], **init_args: Any) -> None:
        self.cls = cls
        self.init_args = init_args

    def __iter__(self) -> Iterator[Any]:
        as_tuple = (self.cls, self.init_args)
        return iter(as_tuple)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        init_args_strings = [
            f"{key}={value!r}" for key, value in self.init_args.items()
        ]
        args_repr = ", ".join([self.cls.__name__] + init_args_strings)
        return f"{class_name}({args_repr})"


class MiddlewareManager:
    def __init__(self, middleware: Optional[Sequence[MiddlewareSpec]], *args, **kwargs):

        self.middleware: Optional[Sequence[MiddlewareSpec]] = []
        self.state = {}
        self._lock = threading.Lock

        self._update_middleware_spec_send(middleware)

    def _update_middleware_spec_send(
        self, middleware: Optional[Sequence[MiddlewareSpec]]
    ):
        if middleware is not None:
            for next_middleware, init_args in middleware:
                next_middleware.send = self.recieve
                self.middleware.append(MiddlewareSpec(next_middleware, **init_args))

    def recieve(self, state: Dict):
        with self._lock():
            for key in state.keys():
                if key not in self.state:
                    self.state[key] = []
                self.state[key].append(state[key])
