import subprocess
from settings import Manager


class Benchmarker():

    def get_benchmarks(
        self, 
        model: str, 
        engine: str, 
        batch: int, 
        time: int, 
        scenario: str
    ):

        model = Manager.models[model]
        engine = Manager.engines[engine]
        
        cmd = [
            f"deepsparse.benchmark {model} \
            --engine {engine} \
            --batch_size {batch} \
            --time {time} \
            --scenario {scenario}"
        ]
        return subprocess.check_output(cmd, shell=True).decode("utf-8")
    

