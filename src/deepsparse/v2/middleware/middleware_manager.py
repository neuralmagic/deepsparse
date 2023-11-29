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


from typing import List, Optional

from .abstract_middleware import AbstractMiddleware


class MiddlewareManager:
    def __init__(self, middleware: Optional[List[AbstractMiddleware]] = None):
        self.middleware = middleware

    def start_event(self, *args, **kwargs) -> None:
        if self.middleware is not None:
            for middleware in self.middleware:
                middleware.start_event(*args, **kwargs)

    def end_event(self, *args, **kwargs) -> None:
        if self.middleware is not None:
            for middleware in self.middleware[::-1]:
                middleware.end_event(*args, **kwargs)
