
import logging
import sqlite3
from typing import Dict
from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject
from deepsparse.dependency_injector.container    import Container
from deepsparse.dependency_injector.services import TimerService

class Pipeline:
    def __init__(self, op: list):
        container = Container()
        container.init_resources()
        container.wire(packages=[__name__, "deepsparse.dependency_injector"])
        self.op = op
        
    @inject
    def  get_timer(
        self,
        timer_service: TimerService = Provide[Container.timer_service],
    ):
        print("pipeline", timer_service)
        id = "pipeline"
        import time
        
        with timer_service.record(id=id):
            time.sleep(1)
            
        print(timer_service.get(id))
        
        c = 0
        for oop in self.op:
            op_time = oop.foo(c)
            c += 1
            
        breakpoint()
        return timer_service
