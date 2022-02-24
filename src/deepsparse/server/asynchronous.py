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


import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional


__all__ = ["initialize_aysnc", "check_initialized", "execute_async"]


_LOOP = None  # type: Optional[asyncio.AbstractEventLoop]
_THREADPOOL = None  # type: Optional[ThreadPoolExecutor]


def initialize_aysnc(max_workers: int = 10):
    global _LOOP, _THREADPOOL

    if _LOOP is not None or _THREADPOOL is not None:
        del _LOOP
        del _THREADPOOL

    _LOOP = asyncio.get_event_loop()
    _THREADPOOL = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="deepsparse.server.async"
    )


def check_initialized():
    if _LOOP is None or _THREADPOOL is None:
        raise RuntimeError(
            "intialize_async must be called first, either _LOOP or _THREADPOOL is None"
        )


async def execute_async(pipeline: Callable, *args, **kwargs):
    check_initialized()
    pipeline = partial(pipeline, *args, **kwargs)
    result = await _LOOP.run_in_executor(_THREADPOOL, pipeline)

    return result
