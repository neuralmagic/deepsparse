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

"""
Description:
Middlwares are used to as an intermediate step for a desired function to carry out any
 necessary logic. Ex. Logging, timing, authentication, ...

Pipeline and Operator uses Middleware, but middlewares logic are not a core
 functionality of Op or Pipeline. That is, Pipeline and Op can run without middlewares,
 and their outputs should be the same using middlewares

Lifecycle with Pipeline using Middleware:
Pipeline -> Middleware -> Pipeline.__call__() -> Middleware

Lifecycle with Pipeline and Ops using Middleware:
Pipeline -> Middleware (from pipeline) -> Pipeline.__call__()
 -> Middleware (from op) -> Op -> Middleware (from op) -> Middleware (from pipeline)

Usage:
Please check tests/deepsparse/v2/middleware
"""


import threading
from typing import Any, Callable, Dict, Iterator, Optional, Protocol, Sequence


class MiddlewareCallable(Protocol):
    """
    Newly created middlewares should inherit this class
    """

    def __call__(self, *args, **kwargs):
        ...

    def send(self, dct: Dict):
        """
        Update middleware Manager state
        Logic defined in MiddlewareManager._update_middleware_spec_send
        """
        ...


class MiddlewareSpec:
    """
    Used to feed the middlewares to the MiddlewareManager
    :param cls: the middleware
    :kwargs init_args: the args used to intialize the cls
    """

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
    """
    A class to manage its state and the middlewares

    Useage:
    middleware_manager = MiddlewareManager( [
        MiddlewareSpec(PrintingMiddleware, identifier="A"), ...]
    )

    :param middleware: List of MiddlewareSpecs
    :param state: state that is shared amongst all the middleware
    :param _lock: lock for the state
    """

    def __init__(self, middleware: Optional[Sequence[MiddlewareSpec]], *args, **kwargs):

        self.middleware: Optional[Sequence[MiddlewareSpec]] = []
        self.state = {}
        self._lock = threading.Lock()

        self._update_middleware_spec_send(middleware)

    def recieve(self, reducer: Callable[[Dict], Dict], *args, **kwargs):
        with self._lock:
            self.state = reducer(self.state, *args, **kwargs)

    def add_middleware(self, middleware: Sequence[MiddlewareSpec]):
        self._update_middleware_spec_send(middleware)

    def _update_middleware_spec_send(
        self, middleware: Optional[Sequence[MiddlewareSpec]]
    ):
        if middleware is not None:
            for next_middleware, init_args in middleware:

                # allow the middleware to communivate with the manager
                next_middleware.send = self.recieve

                self.middleware.append(MiddlewareSpec(next_middleware, **init_args))
