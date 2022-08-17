
from deepsparse.server.pipeline_logger import PipelineLogger

from prometheus_client import start_http_server, Summary

class PrometheusPipelineLogger(PipelineLogger):

    def __init__(self, port: int = 8000):



        self.pre_latency_summary = Summary('pre_latency_seconds', 'Description of summary')



    def log_data(self):
        pass


    def _log_pre_latency(self, t_0, t_1):
        self.pre_latency_summary.observe(t_1-t_0)
