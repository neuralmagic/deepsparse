
import logging
import sqlite3
from typing import Dict
from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject
from deepsparse.dep_inj.container    import Container, UserService


# class BaseService:

#     def __init__(self) -> None:
#         self.logger = logging.getLogger(
#             f"{__name__}.{self.__class__.__name__}",
#         )


# class UserService(BaseService):

#     def __init__(self) -> None:
#         # self.db = db
#         super().__init__()

#     # def get_user(self, email: str) -> Dict[str, str]:
#     #     self.logger.debug("User %s has been found in database", email)
#     #     return {"email": email, "password_hash": "..."}
    
#     def timer(self):
#         return "timer"
    
    


# class Container(containers.DeclarativeContainer):

#     config = providers.Configuration(ini_files=["config.ini"])
#     database_client = providers.Singleton(
#         sqlite3.connect,
#         config.database.dsn,
#     )

#     # Services
#     user_service = providers.Factory(
#         UserService,
#     )
    
class Pipeline:
    def __init__(self, op: list):
        container = Container()
        container.init_resources()
        container.wire(packages=[__name__, "deepsparse.dep_inj"])
        self.op = op
        
    @inject
    def  get_timer(
        self,
        user_service: UserService = Provide[Container.user_service],
    ):
        print("pipeline", user_service)
        
        self.op[0].foo()
        return user_service
