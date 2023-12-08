import logging
import sqlite3
from typing import Dict
from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject


class BaseService:

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}",
        )


class UserService(BaseService):

    def __init__(self) -> None:
        # self.db = db
        super().__init__()

    # def get_user(self, email: str) -> Dict[str, str]:
    #     self.logger.debug("User %s has been found in database", email)
    #     return {"email": email, "password_hash": "..."}
    
    def timer(self):
        return "timer"
    
    


class Container(containers.DeclarativeContainer):

    config = providers.Configuration(ini_files=["config.ini"])
    database_client = providers.Singleton(
        sqlite3.connect,
        config.database.dsn,
    )

    # Services
    user_service = providers.Factory(
        UserService,
    )