
import time
from contextlib import contextmanager
import threading
from typing import Dict

class Timer:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = {}
        self.measurements = {}

    def start(
        self,
        id: str,
    ):
        with self._lock:
            self.start_time[id] = time.time()

    def stop(
        self,
        id: str,
    ):
        end_time = time.time()
        with self._lock:
            if id in self.start_time:
                start_time = self.start_time[id]
                del self.start_time[id]
                self.measurements[id] = end_time - start_time

    @contextmanager
    def record(self, id: str):
        start = time.time()
        yield
        with self._lock:
            self.measurements[id] = time.time() - start

class TimerManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.measurements = {}
        
    def get_new_timer(self):
        return Timer()
    
    def update(self, measurements: Dict[str, float]):
        with self.lock:
            self.measurements.update(measurements)