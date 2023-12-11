from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject
from deepsparse.dep_inj.container    import Container, UserService, TimerService

import time
class Op:
    def __init__(self):
        ...
    
    @inject
    def foo(
        self, 
            user_service: UserService = Provide[Container.user_service],
            timer_service: TimerService = Provide[Container.timer_service],):
        print("Op")
        id = "op"
        
        timer_service.start(id)
        time.sleep(0.2)
        timer_service.end(id)
        print(timer_service.get(id))
        return user_service
        ...