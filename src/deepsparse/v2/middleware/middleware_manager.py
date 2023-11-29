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


from typing import List


class MiddlewareManager:
    def __init__(self, middlewares: List):
        self.middlwares = {}
        self._init(middlewares)

    def _init(self, middleware: List):
        for middleware in self.middlewares:
            self.middlwares[middleware.__name__] = middleware

    def start_event(self, name: str, *args, **kwargs):
        self.middlwares[name].start_event(*args, **kwargs)

    def end_event(self, name: str, *args, **kwargs):
        self.middlwares[name].start_event(*args, **kwargs)
