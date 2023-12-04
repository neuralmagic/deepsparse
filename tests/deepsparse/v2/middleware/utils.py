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


from typing import Any, Dict

from deepsparse.v2.middleware.middlewares import MiddlewareCallable


class DummyMiddleware(MiddlewareCallable):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        ...

    def __call__(self, *args, **kwargs) -> Any:
        ...


class ReducerMiddleware(DummyMiddleware):
    def __call__(self, *args, **kwargs) -> Any:
        self.send(self.reducer, 1)

    def reducer(self, state: Dict, *args, **kwargs):
        name = self.__class__.__name__
        if name not in state:
            state[name] = []
        state[name].append(args[0])
        return state


class PrintingMiddleware(MiddlewareCallable):
    def __init__(self, call_next: MiddlewareCallable, identifier: str):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:
        print(f"{self.identifier}: before call_next")
        result = self.call_next(*args, **kwargs)
        print(f"{self.identifier}: after call_next: {result}")
        return result


class SendStateMiddleware(MiddlewareCallable):
    def __init__(self, call_next: MiddlewareCallable, identifier: str):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:
        name = self.__class__.__name__
        self.send(self.reducer, 0)

        result = self.call_next(*args, **kwargs)
        self.send(self.reducer, 1)

        return result

    def reducer(self, state: Dict, *args, **kwargs):
        name = self.__class__.__name__
        if name not in state:
            state[name] = []
        state[name].append(args[0])
        return state
