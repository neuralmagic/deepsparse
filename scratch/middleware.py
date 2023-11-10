####

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



"""
dispatch event need to add error handling. Whats the paln for that
exception therown, break the whole thing or resilient?

"""

from deepsparse.v2.pipeline import Pipeline
from typing import List, Callable, Union, Any

class MiddlewarePipeline(Pipeline):
    def __init__(self, middleware_specs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.middleware: List = [
            middle_spec.build(self) for middle_spec  in middleware_specs
        ]
        self.dispatch_event("init")
        
    def add_middleware(self, middleware):
        self.middleware.append(middleware)
    
    
    def dispatch_event(self, event_name: str, ):
        for middleware in self.middleware:
            middleware(event_name)
            
    def run(
        self,
        *args,
        inference_state,
        pipeline_state,
        **kwargs,
    ):
        self.dispatch_event("run_start")
        inp = super().run(
            *args,
            inference_state=inference_state,
            pipeline_state=pipeline_state,
            **kwargs,
        )
        self.dispatch_event("run_end")
        return inp
        
        
    def _run_next_step(
        self,
        *args,
        func: Callable,
        next_step: Union[str, int],
        input: Any = None,
        **kwargs,
    ):
        self.dispatch_event("op_start")
        inp = super()._run_next_step(
        *args,
        func=func,
        next_step=next_step,
        input=input,
        **kwargs,)
        self.dispatch_event("op_end")

        return inp
        
        
class BaseMiddleware:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        
    def __call__(self, event_name: str):
        ...

        
class MiddlewareSpec:
    def __init__(self, middleware_class, *args, **kwargs):
        self.middleware_class = middleware_class
        self.args = args
        self.kwargs = kwargs

    def build(self, pipeline):
        return self.middleware_class(
            pipeline, 
            *self.args, 
            **self.kwargs
        )
        
class MiddlewareTimer(BaseMiddleware):
    
    def __call__(self, event_name):
        print(event_name)




        
m_specs = [MiddlewareSpec(MiddlewareTimer)]
kwargs = {
    "ops": [AddOneOperator(), AddTwoOperator()],
    "router": LinearRouter(end_route=2),
    "schedulers":[OperatorScheduler()],
}


AddThreePipelineMiddleware = MiddlewarePipeline(m_specs, **kwargs)

AddThreePipeline = Pipeline(
    ops=[AddOneOperator(), AddTwoOperator()],
    router=LinearRouter(end_route=2),
    schedulers=[OperatorScheduler()],
)

pipeline_input = IntSchema(value=5)
# pipeline_output = AddThreePipeline(pipeline_input)
pipeline_output = AddThreePipelineMiddleware(pipeline_input)


print(pipeline_output.value)
