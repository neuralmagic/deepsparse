from typing import Dict

from pydantic import BaseModel

from deepsparse.v2 import Pipeline
from deepsparse.v2.operators import Operator
from deepsparse.v2.routers import LinearRouter
from deepsparse.v2.schedulers import OperatorScheduler
# from deepsparse.v2.utils import Context
# from deepsparse.v2.timer import Timer


# from src.deepsparse.v2.timed_pipeline  import time



class IntSchema(BaseModel):
    value: int


class AddOneOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 1}


class AddTwoOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 2}



AddThreePipeline = Pipeline(
    ops=[AddOneOperator(), AddTwoOperator()],
    router=LinearRouter(end_route=2),
    schedulers=[OperatorScheduler()],
)

# timer = Timer()
# AddThreePipeline.timer = timer

# timed_pipeline = time(AddThreePipeline)


pipeline_input = IntSchema(value=5)
pipeline_output = AddThreePipeline(pipeline_input)
# pipeline_output = timed_pipeline(pipeline_input)

print(pipeline_output.value)

# print(r)

assert pipeline_output.value == 8
breakpoint()
"""
(Pdb) AddThreePipeline.timer_middleware.timer.measurements
{'init': 1.001140832901001, '0': 0.0006783008575439453, '1': 0.0001842975616455078}
"""




"""
python3 -m scratch.pipeline_v2
python -m profile -s time scratch/pipeline_v2.py
"""
