from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject
from deepsparse.dependency_injector.container    import Container
from deepsparse.dependency_injector.services import TimerService


import time
class Op:
    def __init__(self):
        ...
    
    @inject
    def foo(
        self, 
        counts,
        timer_service: TimerService = Provide[Container.timer_service],):
        print("Op")
        id = "op" + str(counts)
        with timer_service.record(id=id):
            time.sleep(0.2)

        print(timer_service.get(id))