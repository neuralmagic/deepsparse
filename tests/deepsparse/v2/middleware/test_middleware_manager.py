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

from src.deepsparse.v2.middleware import MiddlewareCallable, MiddlewareManager


class PrintingMiddleware(MiddlewareCallable):
    def __init__(self, app: MiddlewareCallable, identifier: str):
        self.identifier: str = identifier
        self.app: MiddlewareCallable = app

    def __call__(self, *args, **kwargs) -> Any:
        print(f"{self.identifier}: before app")
        result = self.app(*args, **kwargs)
        print(f"{self.identifier}: after app: {result}")
        return result


class BaseApp:
    def __call__(self, *args, **kwargs):
        print(f"BASE APP!\n  args: {args}\n  kwargs: {kwargs}")


def test_dummy_example():
    subject = MiddlewareManager(
        middleware=[
            [PrintingMiddleware, "A"],
            [PrintingMiddleware, "B"],
            [PrintingMiddleware, "C"],
        ]
    )
    base_app = BaseApp()
    subject.wrap(base_app, 1, 2, 3, a="a", b="b", c="c")
