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


from typing import Any, Optional, Sequence


from deepsparse.v2.middleware.middleware_spec import MiddlewareCallable, MiddlewareSpec


class MiddlewareManager:
    def __init__(
        self,
        middleware: Optional[Sequence[MiddlewareSpec]],
    ):
        self.middleware: Optional[Sequence[MiddlewareSpec]] = middleware

    def wrap(self, base_app: MiddlewareCallable, *args, **kwargs) -> Any:
        app = base_app
        if self.middleware is not None:
            for middleware, init_args in reversed(self.middleware):
                app = middleware(app, **init_args)
        return app(*args, **kwargs)
