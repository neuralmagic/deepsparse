from queue import Queue
from typing import List
from config import MODEL_PORT
import numpy as np
import time, requests, json
from sklearn.preprocessing import normalize

class SentenceModel:
    def __init__(self, 
        model_host="localhost",
        timing=True
    ):
        self._model_url = f"http://{model_host}:{MODEL_PORT}/predict"
        self._timing = timing
        if self._timing:
            self._time_queue = Queue()

    def make_inference_request(self, data:List[str]):
        obj = {
            'inputs': data
        }
        response = requests.post(self._model_url, json=obj)
        return json.loads(response.text)["embeddings"]

    def sentence_encode(self, data:List[str], is_load=False):
        start = time.perf_counter()
        embedding = self.make_inference_request(data)
        sentence_embeddings = normalize(np.array(embedding)).tolist()
        end = time.perf_counter()

        if self._timing and not is_load:
            self._time_queue.put([start, end])
        
        return sentence_embeddings

    def compute_latency(self):
        batch_times = list(self._time_queue.queue)
        if len(batch_times) == 0:
            return {
                "msg" : "Latency data has been cleared"
            }

        batch_times_ms = [
            (batch_time[1] - batch_time[0]) * 1000 for batch_time in batch_times
        ]
                
        self._time_queue.queue.clear()
        
        return {
            "count" : len(batch_times),
            "median": np.median(batch_times_ms),
            "mean": np.mean(batch_times_ms),
            "std": np.std(batch_times_ms)
        }