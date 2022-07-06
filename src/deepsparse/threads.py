import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


class ThreadPool:
    def __init__(
        self,
        workers=10,
        thread_name_prefix="deepsparse.pipelines.async",
        *args,
        **kwargs
    ):
        self._event_loop = asyncio.get_event_loop()
        self._threadpool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=thread_name_prefix,
            *args,
            **kwargs,
        )

    def submit(self, job):
        assert callable(job)
        return self._threadpool.submit(
            job
        )

