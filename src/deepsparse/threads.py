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
from concurrent.futures import Future, ThreadPoolExecutor


class ThreadPool:
    """
    A generic ThreadPool class for DeepSparse, callable jobs can be
    submitted to this class.

    Usage:
    >>> from time import sleep
    >>> from deepsparse import ThreadPool
    >>> t = ThreadPool()
    >>> future = t.submit(lambda : sleep(1)) # non-blocking call
    >>> future.result() # blocking call

    :param workers: Maximum number of workers to use in this threadpool,
        defaults to 10
    :param thread_name_prefix: the prefix name for each new thread created in the
        ThreadPool, deafaults to `deepsparse.thread`
    :param args: extra positional arguments for initializing ThreadPoolExecutor
    :param kwargs: extra keyword args for initializing ThreadPoolExecutor
    """

    def __init__(
        self,
        workers: int = 10,
        thread_name_prefix: str = "deepsparse.thread",
        *args,
        **kwargs,
    ):
        self._event_loop = asyncio.get_event_loop()
        self._threadpool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=thread_name_prefix,
            *args,
            **kwargs,
        )

    def submit(self, job) -> Future:
        """
        Utility method to submit jobs to the ThreadPool

        :pre: The job must be a callable
        :param job: Task to be executed in a Thread
        :return: A Future object, to get the result use Future.result(), note:
            Future.result() is a blocking call
        """
        assert callable(job)
        return self._threadpool.submit(job)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
