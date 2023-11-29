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


from typing import Any, Iterator, Protocol


class MiddlewareCallable(Protocol):
    def __call__(self, *args, **kwargs):
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
