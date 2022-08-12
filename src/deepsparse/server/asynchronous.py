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


"""
Threadpools and logic to handle asynchronous requests for pipelines using
the DeepSparse Engine. Used to enable multiple requests when serving models.
"""


import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional


__all__ = ["initialize_async", "check_initialized", "execute_async"]


_LOOP = None  # type: Optional[asyncio.AbstractEventLoop]
_THREADPOOL = None  # type: Optional[ThreadPoolExecutor]


def initialize_async(max_workers: int = 10):
    """
    Initialize the async loop and threadpool for the execute_async function

    :param max_workers: the number of maximum workers that can run at one time
    """
    global _LOOP, _THREADPOOL

    if _LOOP is not None or _THREADPOOL is not None:
        del _LOOP
        del _THREADPOOL

    _LOOP = asyncio.get_event_loop()
    _THREADPOOL = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="deepsparse.server.async"
    )


def check_initialized():
    """
    Check that initialize_async has been called, if not raise RuntimeError
    """
    if _LOOP is None or _THREADPOOL is None:
        raise RuntimeError(
            "intialize_async must be called first, either _LOOP or _THREADPOOL is None"
        )


async def execute_async(pipeline: Callable, *args, **kwargs):
    """
    Execute a pipeline or engine request using the given args and kwargs asynchronously.

    :param pipeline: a callable pipeline or engine execution
    :param args: the positional arguments to pass into the pipeline
    :param kwargs: the keyword arguments to pass into the pipeline
    :return: the result of pipeline(*args, **kwargs) after it has run asynchronously
    """
    check_initialized()
    pipeline = partial(pipeline, *args, **kwargs)
    result = await _LOOP.run_in_executor(_THREADPOOL, pipeline)

    return result
