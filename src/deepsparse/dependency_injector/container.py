import logging
import sqlite3
from typing import Dict, Optional
from dependency_injector import containers, providers


from dependency_injector.wiring import Provide, inject
import threading
import time

from deepsparse.dependency_injector.services import TimerService

    
class Container(containers.DeclarativeContainer):
    timer_service = providers.Singleton(
        TimerService,
    )