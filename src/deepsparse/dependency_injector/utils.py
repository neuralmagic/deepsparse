

import threading
import time


class Timer:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = {}
        self.measurements = {}

    def start_event(
        self,
        name: str,
    ):
        with self._lock:
            self.start_time[name] = time.time()

    def end_event(
        self,
        name: str,
    ):
        end_time = time.time()
        with self._lock:
            if name in self.start_time:
                start_time = self.start_time[name]
                del self.start_time[name]
                self.measurements[name] = end_time - start_time