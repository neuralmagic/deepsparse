import time
from functools import wraps
import threading

class Timer:
    
    _lock = threading.Lock()
    condition = threading.Condition()
    
    def __init__(self):
        with self._lock:
            self.measurements = {}

    def start(self, key):
        with self._lock:
            self.measurements[key] = time.time()

    def end(self, key):
        end_time = time.time()
        
        if key in self.measurements:
            start_time = self.measurements[key]
            del self.measurements[key]
            return end_time - start_time
        return None

    def _timeout(self, wait: int=10):
        time.wait(wait)
        return True


    def clock(timer, func):
        # breakpoint()
        @wraps(func)
        def wrapper(self, inp, context):
            start_time = time.time()

            # Call the original method
            result = func(self, inp, context)

            end_time = time.time()
            runtime = end_time - start_time

            # Store the runtime with class name and function name as the key
            key = f"{self.__class__.__name__}.{func.__name__}"
            breakpoint()
            timer.measurements[key] = runtime

            return result
        breakpoint()
        return wrapper


"""

Wall time vs cpu tume

"""