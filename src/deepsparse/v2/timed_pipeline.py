from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.timer import Timer


from pydantic import BaseModel

from deepsparse.v2 import Pipeline
from deepsparse.v2.operators import Operator
from deepsparse.v2.routers import LinearRouter
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.utils import Context
from typing import Dict



class TimedPipeline(Pipeline):

    def __init__(
            self,  
            *args,       
            **kwargs,
        ):
        
        self._timer = Timer()
        super().__init__(*args,**kwargs)
    
    # def run(self, *args, **kwargs):
    def run(self, inp, context):
        # breakpoint()
        self._timer.start("nar")
        # breakpoint()
        r = super().run(**{"inp":inp, "context":context})
        self._timer.end("nar")
        
        return r
    
    def evaludate_operator(self, *args, **kwargs):
        # breakpoint()
        self._timer.start("goo")
        r = super().evaluate_operator(*args, **kwargs)
        self._timer.end("goo")
        return r 


def time(p: Pipeline,):
    t = TimedPipeline(
        ops=[p],
        router = LinearRouter(end_route=1),
        schedulers = p.schedulers
    )

    return t