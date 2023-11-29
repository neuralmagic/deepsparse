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
        self.middlewares = {}
        self._init(middlewares)

    def _init(self, middleware: List):
        for middleware in middleware:
            init_middleware = middleware()
            self.middlewares[init_middleware.__name__] = init_middleware

    def start_event(self, name: str, *args, **kwargs):
        self.middlewares[name].start_event(*args, **kwargs)

    def end_event(self, name: str, *args, **kwargs):
        self.middlewares[name].end_event(*args, **kwargs)

    def dispatch(self, command: str, *args, **kwargs):
        """
        command should include both the middleware
         key and the function inside the selected middleware

        dispatch("timer.update_middleware_timer", *args, **kwargs)
        """
        name, func_name = command.split(".")
        if name in self.middlewares:
            target_func = self.middlewares[name]
            method = getattr(target_func, func_name, None)
            if callable(method):
                return method(*args, **kwargs)

    def __getitem__(self, key):
        return self.middlewares[key]

    def iterate_dict(self):
        for key in self.middlewares:
            yield key, self.middlewares[key]
