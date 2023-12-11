
import logging
import sqlite3
from typing import Dict
from dependency_injector import containers, providers

from dependency_injector.wiring import Provide, inject

from deepsparse.dependency_injector.container    import Container
from deepsparse.dependency_injector.pipeline import Pipeline
from deepsparse.dependency_injector.ops import Op


print(0)

op = [Op(), Op()]

p = Pipeline(op=op)

timer_service = p.get_timer()
print(timer_service)
print(timer_service.timer.measurements)


class MockOp(Op):
    def foo():
        mocker.patch("")
        ...
    
