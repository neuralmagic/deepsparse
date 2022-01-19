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
Exposes a convenience function to add queuing and multi-threaded
consumption capabilities to Pipeline object from deepsparse.transformers
module
"""
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Optional

from deepsparse.transformers import Pipeline, pipeline

from .schemas import REQUEST_MODELS, RESPONSE_MODELS
from .utils import PipelineEngineConfig


__all__ = [
    "get_throttled_engine_pipeline",
    "get_request_model",
    "get_response_model",
    "ThrottleWrapper",
]


class ThrottleWrapper:
    """
    A Throttle Wrapper over DeepSparse ENGINE. Should not be instantiated
    directly, use `get_throttled_pipeline(...)` to maintain a single
    copy of the ENGINE. This wrapper limits the maximum number of concurrent
    calls to the callable_func with extra calls waiting in a queued _threadpool

    :param engine_callable: A Callable func or class to add throttling
            capabilities to
    :param max_workers: An integer representing max concurrent
        consumption limit
    """

    def __init__(
        self,
        engine_callable: Callable,
        max_workers: int = 3,
    ):
        self._engine: Callable = engine_callable
        self._max_workers: int = max_workers
        self._threadpool = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="deepsparse_engine_worker"
        )

    def __call__(self, *args, **kwargs) -> Future:
        """
        Submits a task to the callable with throttled speed, and returns
        corresponding Future object

        :return: Future object corresponding to the submitted task. The result
            can be fetched using Future.result() Note: calling result() on the
            returned object will be a blocking call
        """
        future = self._threadpool.submit(self._engine, *args, **kwargs)
        return future

    @property
    def max_workers(self) -> int:
        """
        :return: max number of concurrent workers
        """
        return self._max_workers


@lru_cache()
def get_throttled_engine_pipeline(
    config: Optional[PipelineEngineConfig] = None,
) -> Optional[ThrottleWrapper]:
    """
    Factory method to get a throttled Pipeline ENGINE, based on config read
    from the environment. Recommended safe way to instantiate ThrottleWrapper,
    (maintains a single copy of the backend deepsparse engine)

    :param config: A PipelineEngineConfig object to create throttled pipeline.
        if None the config is read from the environment.
    :return: ThrottleWrapper callable object that wraps Pipeline from
        deepsparse.transformers module
    """
    _pipeline_config: PipelineEngineConfig = config or PipelineEngineConfig.get_config()
    if _pipeline_config.task is None:
        return None
    _pipeline_engine: Pipeline = pipeline(
        model_path=_pipeline_config.model_file_or_stub,
        task=_pipeline_config.task,
        num_cores=_pipeline_config.num_cores,
        scheduler=_pipeline_config.scheduler,
        max_length=_pipeline_config.max_length,
    )
    return ThrottleWrapper(
        engine_callable=_pipeline_engine,
        max_workers=_pipeline_config.concurrent_engine_requests,
    )


@lru_cache()
def get_request_model():
    """
    :return: Request Model Schema
    """
    _pipeline_config: PipelineEngineConfig = PipelineEngineConfig.get_config()
    return REQUEST_MODELS[_pipeline_config.task]


@lru_cache()
def get_response_model():
    """
    :return: Response Model Schema
    """
    _pipeline_config: PipelineEngineConfig = PipelineEngineConfig.get_config()
    return RESPONSE_MODELS[_pipeline_config.task]
