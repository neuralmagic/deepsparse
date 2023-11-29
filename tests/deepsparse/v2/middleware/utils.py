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


from deepsparse.v2.middleware import AbstractMiddleware


class OpsTrackerMiddleware(AbstractMiddleware):
    def __init__(self):
        self.start_order = []
        self.end_order = []

    def start_event(self, *args, **kwargs):
        self.start_order.append(kwargs.get("name"))

    def end_event(self, *args, **kwargs):
        self.end_order.append(kwargs.get("name"))


class CounterMiddleware(AbstractMiddleware):
    def __init__(self):
        self.start_called = 0
        self.end_called = 0

    def start_event(self, *args, **kwargs):
        self.start_called += 1

    def end_event(self, *args, **kwargs):
        self.end_called += 1
