"""
Implementation of the Timing Logger that collects the time delta
elapsed between the start and end of an event
"""
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel
from collections import defaultdict
from deepsparse.timing import InferencePhases
from deepsparse.loggers import BaseLogger, MetricCategories


__all__ = ["TimingLogger"]

class TimeMeasurement(BaseModel):
    """
    Data structure that holds the time measurement information
    """
    identifier: str
    value: float
    batch_size: Optional[int]

class TimingLogger(BaseLogger):
    """
    Timing Logger that collects and aggregates the time deltas
    elapsed between the start and end of an event
    """

    time_measurement = []

    # get attributes of dataclass InferencePhases as list
    atts =


    def log(self, identifier: str, value: Any, category: MetricCategories, **kwargs):
        """
        Collect information from the pipeline and pipe it them to the stdout

        :param identifier: The name of the item that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        :param kwargs: Additional keyword arguments to pass to the logger
        """
        self.time_measurement.append(TimeMeasurement(identifier, value, kwargs.get("batch_size")))

    def __str__(self):
        return "TimingLogger" # TODO