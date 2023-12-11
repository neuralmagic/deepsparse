import logging
import sqlite3
from typing import Dict, Optional
from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject
import threading


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
    
    
import time
class Timer:
    def __init__(self):
        self._lock = threading.Lock()
        self.start_time = {}
        self.measurements = {}

    def start_event(
        self,
        name: str,
        inputs: Optional[Dict] = None,
        inference_state: Optional["InferenceState"] = None,
        **kwargs,
    ):
        with self._lock:
            self.start_time[name] = time.time()

    def end_event(
        self,
        name: str,
        inputs: Optional[Dict] = None,
        inference_state: Optional["InferenceState"] = None,
        **kwargs,
    ):
        end_time = time.time()
        with self._lock:
            if name in self.start_time:
                start_time = self.start_time[name]
                del self.start_time[name]
                self.measurements[name] = end_time - start_time
    
class TimerService(BaseService):
    def __init__(self):
       self.timer = Timer()
        
    def start(self, id: str):
        self.timer.start_event(id)
        
    def end(self, id: str):
        self.timer.end_event(id)
        
    def get(self, id: str): # get in measurements
        return self.timer.measurements[id]
        
    def exist(self, id: str): # check if in start
        ...
    
    
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
    timer_service = providers.Factory(
        TimerService,
    )