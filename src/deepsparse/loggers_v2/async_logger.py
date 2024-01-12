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


import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, Callable, Optional


class AsyncLogger:
    def __init__(self, max_workers=1, **config):
        self._job_pool: Executor = ThreadPoolExecutor(max_workers=max_workers)
        self.config = config

    # def log(self, identifier: str, value: Any, category: MetricCategories, **kwargs):
    def log(self, logger: Callable, value: Any, tag: Optional[str] = None, **kwargs):

        """
        Forward log calls to wrapped logger to run asynchronously

        :param identifier: The name of the item that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        :param kwargs: Additional keyword arguments to pass to the logger
        """
        job_future = self._job_pool.submit(
            logger.log,
            value=value,
            # tag=tag,
            **kwargs,
        )
        job_future.add_done_callback(self._log_async_job_exception)

    def _log_async_job_exception(self, future):
        exception = future.exception()
        if exception is not None:
            self.logger.error(
                f"Exception occurred during async logging job: {repr(exception)}"
            )

    def setup_file_handlers(logger, config):
        """Set up handlers for the given python logger"""
        if filename := config.get("filename"):
            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(config.get("level", logging.INFO))
            formatter = logging.Formatter(
                config.get("formatter", "%(asctime)s - %(levelname)s - %(message)s")
            )
            file_handler.setFormatter(formatter)

            # Add the FileHandler to the logger
            logger.addHandler(file_handler)
