from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject
from deepsparse.dep_inj.container    import Container, UserService


class Op:
    def __init__(self):
        ...
    
    @inject
    def foo(self, user_service: UserService = Provide[Container.user_service],):
        print("Op")
        return user_service
        ...