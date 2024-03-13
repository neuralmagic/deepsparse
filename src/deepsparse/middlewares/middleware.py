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
    Middlwares are used to as an intermediate step for a desired function to
     carry out anynecessary logic. Ex. Logging, timing, authentication, ...
    Pipeline and Operator use Middleware, but middlewares logic is not a core
     functionality of Operator or Pipeline. That is, Pipeline(s) and Operator(s)
     can run without middlewares, and their outputs should be the same
     using middlewares

Lifecycle:
    Vanilla Pipeline:
        Pipeline -> Pipeline.__call__() -> Pipeline.run() -> ...

    Pipeline + Middleware:
        Pipeline -> Pipeline.__call__()
            -> Middleware
                -> Pipeline.run()
            -> Middleware
        -> ...

     Pipeline + Middleware, Operator + Middleware:
        Pipeline -> Pipeline.__call__()
            -> Middleware (for pipeline start)
                -> Pipeline.run()
                    -> Middleware (for operator start)
                        -> Operator.run()
                    -> Middleware (for operator end)
            -> Middleware (for pipeline end)
        -> ...

Usage:
    Please check tests/deepsparse/middlewares
"""


import threading
from typing import Any, Callable, Dict, Iterator, Optional, Protocol, Sequence, Type

from deepsparse.operators import Operator


class MiddlewareCallable(Protocol):
    """
    Newly created middlewares should inherit this class
    """

    def __call__(self, *args, **kwargs):
        """
        Pipeline, Operator callable will be overwritten
        and wrapped with Middleware callable
        """

    def send(self, reducer: Callable[[Dict], Dict]):
        """
        Update middleware Manager state
        Logic defined in MiddlewareManager._update_middleware_spec_send

        :param reducer: A callable that contains logic to update
         the middleware state
        """


class MiddlewareSpec:
    """
    Used to feed the middlewares to the MiddlewareManager

    :param cls: the middleware
    :kwargs init_args: the args used to intialize the cls
    """

    def __init__(self, cls: Type[MiddlewareCallable], **init_args: Dict) -> None:
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

    def __init__(
        self, middleware: Optional[Sequence[MiddlewareSpec]] = None, *args, **kwargs
    ):

        self.middleware: Optional[
            Sequence[MiddlewareSpec]
        ] = []  # user defined middlewre
        self.state = {}
        self._lock = threading.Lock()

        self._update_middleware_spec_send(middleware)

    def recieve(self, reducer: Callable[[Dict], Dict], *args, **kwargs):
        """
        Call the reducer with the given *args, **kwargs
        Used to write to the state, and called from the middleware

        :param reducer: a callable defined inside the middleware
        """
        with self._lock:
            self.state = reducer(self.state, *args, **kwargs)

    def add_middleware(self, middleware: Sequence[MiddlewareSpec]):
        self._update_middleware_spec_send(middleware)

    def build_middleware_stack(self, next_call: Callable) -> Callable:
        """
        Instantiate the middleware and the last call should be the
        supplied next_call in the signature

        :param next_call: Callable to wrap the middleware to
        """
        if self.middleware is not None:
            for middleware, init_args in reversed(self.middleware):
                next_call = middleware(next_call, **init_args)
        return next_call

    def wrap(self, operator: Operator) -> Callable:
        """
        Add middleware to the operator

        :param operator: the desired operator to wrap middleware to
        """
        wrapped_operator = self.build_middleware_stack(operator)
        return wrapped_operator

    def _update_middleware_spec_send(
        self, middleware: Optional[Sequence[MiddlewareSpec]]
    ):
        """
        Add the recieve function to middleware send function. Used as a way for the
        middleware to write to the manager state

        :param middleware: Optional middleware to add the logic to
        """
        if middleware is not None:
            for next_middleware, init_args in middleware:

                # allow the middleware to communicate with the manager
                next_middleware.send = self.recieve

                self.middleware.append(MiddlewareSpec(next_middleware, **init_args))

    @property
    def middlewares(self):
        return [middleware.cls for middleware in self.middleware]
