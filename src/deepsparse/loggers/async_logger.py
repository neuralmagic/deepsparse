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
Logger wrapper to run log calls asynchronously to not block the main process
"""

import logging
import textwrap
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

from deepsparse.loggers import BaseLogger, MetricCategories


__all__ = ["AsyncLogger"]


_LOGGER = logging.getLogger(__name__)


class AsyncLogger(BaseLogger):
    """
    Logger wrapper that forwards log calls to run asynchronously, freeing
    the main process. No call back/returned future currently provided

    :param logger: logger object to wrap
    :param max_workers: maximum logging tasks to run in the job pool at once.
        defaults to 1
    """

    def __init__(self, logger: BaseLogger, max_workers: int = 1):
        self.logger = logger
        self._job_pool: Executor = ThreadPoolExecutor(max_workers=max_workers)

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        Forward log calls to wrapped logger to run asynchronously

        :param identifier: The name of the item that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        job_future = self._job_pool.submit(
            self.logger.log,
            identifier=identifier,
            value=value,
            category=category,
        )
        job_future.add_done_callback(_log_async_job_exception)

    def __str__(self):
        child_str = textwrap.indent(str(self.logger), prefix="  ")
        return f"{self.__class__.__name__}:\n{child_str}"


def _log_async_job_exception(future):
    exception = future.exception()
    if exception is not None:
        _LOGGER.error(f"Exception occurred during async logging job: {repr(exception)}")
