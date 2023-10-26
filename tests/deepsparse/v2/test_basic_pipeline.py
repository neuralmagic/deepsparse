"""
Simple example and test of a dummy pipeline
"""

from pydantic import BaseModel

from deepsparse.v2 import Pipeline
from deepsparse.v2.operators import Operator
from deepsparse.v2.routers import Router
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.utils import Context, OperatorSchema


class IntSchema(BaseModel):
    value: int


class AddOneOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, context: Context) -> OperatorSchema:
        return IntSchema(value=inp.value + 1)


class AddTwoOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, context: Context) -> OperatorSchema:
        return IntSchema(value=inp.value + 2)


AddThreePipeline = Pipeline(
    stages=[AddOneOperator(), AddTwoOperator()],
    router=Router(),
    schedulers=[OperatorScheduler()],
)


def test_run_simple_pipeline():
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8
