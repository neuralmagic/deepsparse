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


class AbstractMiddleware:
    """All middleware should inhertic"""

    ...

    def __init__(self):
        self._populate()

    def _populate(self):
        # all classes inheriting it class should be mapped
        #
        # some thing like
        #
        # middlewares = [AbstractMiddleware]
        # for klass in middlewares∑
        # self.middlewares = { klass.__name__ : middleware}
        ...
