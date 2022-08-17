
from abc import ABC, abstractmethod
from typing import Callable
class PipelineLogger(ABC):

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def get_logger(cls, pipeline_name: str):
        """
        Factory method to create a PipelineLogger instance.
        PipelinerLogger instance will be related to the pipeline
        by a common name (passed as an argument `pipeline_name`).
        :param pipeline_name:
        :return:
        """
        raise NotImplementedError("")

    def log_latency(self, latency_callback: Callable):
        """

        :param latency_callback:
        :return:
        """

    def log_data(self):
        """

        :return:
        """

    def log_pre_latency(self):
        pass

    def log_post_latency(self):
        pass

    def log_inference_latency(self):
        pass

    def log_total_latency(self):
        pass





