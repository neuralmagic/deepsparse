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


from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Callable, Optional


class AsyncSubmitter:
    def __init__(self, max_workers=1, **config):
        self._job_pool: Executor = ThreadPoolExecutor(max_workers=max_workers)
        self.config = config

    def submit(
        self, func: Callable, callback: Optional[Callable] = None, /, *args, **kwargs
    ):
        job_future = self._job_pool.submit(
            func,
            *args,
            **kwargs,
        )
        if callback is not None:
            job_future.add_done_callback(callback)
