
from dependency_injector.wiring import Provide, inject
import threading
import time
from contextlib import contextmanager
import logging
from typing import Dict, Optional
from dependency_injector import containers, providers

from deepsparse.dependency_injector.utils import Timer

class BaseService:
    def __init__(self) -> None:
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}",
        )

class TimerService(BaseService):
    def __init__(self):
       self.timer = Timer()
        
    def start(self, id: str):
        self.timer.start_event(id)
        
    def end(self, id: str):
        self.timer.end_event(id)
        
    def get(self, id: str):
        return self.timer.measurements[id]
    
    def measurements(self):
        return self.timer.measurements
    
    def start_times(self):
        return seld.timer.start_times
    
    @contextmanager
    def record(self, id: str):
        self.start(id)
        yield
        self.end(id)