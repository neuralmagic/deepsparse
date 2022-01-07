"""
Exposes a convenience function to add queuing and multi-threaded
consumption capabilities to Pipeline object from deepsparse.transformers
module
"""
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable
from functools import lru_cache

from deepsparse.transformers import Pipeline, pipeline

from .schemas import request_models, response_models
from .utils import PipelineConfig

__all__ = [
    'get_throttled_pipeline',
    'get_request_model',
    'get_response_model',
    'ThrottleWrapper',
]


class ThrottleWrapper:
    """
    A Throttle Wrapper over DeepSparse Engine. Should not be instantiated
    directly, use `get_throttled_pipeline(...)` to maintain a single
    copy of the engine. This wrapper limits the maximum number of concurrent
    calls to the callable_func with extra calls waiting in a queued _threadpool

    :param callable_func: A Callable func or class to add throttling
            capabilities to
    :param max_workers: An integer representing max concurrent
        consumption limit
    """

    def __init__(self, callable_func: Callable,
                 max_workers: int = 3,
                 ):
        self._engine: Callable = callable_func
        self._max_workers: int = max_workers
        self._threadpool = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="deepsparse_engine_worker"
        )

    def __call__(self, *args, **kwargs) -> concurrent.futures.Future:
        """
        Submits a task to the callable with throttled speed, and returns
        corresponding Future object

        :return: Future object corresponding to the submitted task. The result
            can be fetched using Future.result() Note: calling result() on the
            returned object will be a blocking call
        """
        future = self._threadpool.submit(
            self._engine, *args, **kwargs
        )
        return future

    @property
    def max_workers(self) -> int:
        """
        :return: max number of concurrent workers
        """
        return self._max_workers


@lru_cache()
def get_throttled_pipeline(
        config: Optional[PipelineConfig] = None
) -> ThrottleWrapper:
    """
    Factory method to get a throttled Pipeline Engine, based on config read
    from the environment. Recommended safe way to instantiate ThrottleWrapper,
    (maintains a single copy of the engine)

    :param config: A PipelineConfig object to create throttled pipeline.
        if None the config is read from the environment.
    :return: ThrottleWrapper callable object that wraps Pipeline from
        deepsparse.transformers module
    """
    _pipeline_config: PipelineConfig = config or PipelineConfig.get_config()
    _pipeline_engine: Pipeline = pipeline(
        model_path=_pipeline_config.model_file_or_stub,
        task=_pipeline_config.task,
        num_cores=_pipeline_config.num_cores,
        scheduler=_pipeline_config.scheduler,
        max_length=_pipeline_config.max_length,
    )
    return ThrottleWrapper(
        callable_func=_pipeline_engine,
        max_workers=_pipeline_config.concurrent_engine_requests,
    )


@lru_cache()
def get_request_model():
    """
    :return: Request Model Schema
    """
    _pipeline_config: PipelineConfig = PipelineConfig.get_config()
    return request_models[_pipeline_config.task]


@lru_cache()
def get_response_model():
    """
    :return: Response Model Schema
    """
    _pipeline_config: PipelineConfig = PipelineConfig.get_config()
    return response_models[_pipeline_config.task]
