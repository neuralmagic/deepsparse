# # Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing,
# # software distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """
# Simple example and test of a dummy pipeline
# """

# from typing import Dict

# from pydantic import BaseModel

# from deepsparse.v2 import Pipeline
# from deepsparse.v2.operators import Operator
# from deepsparse.v2.routers import LinearRouter
# from deepsparse.v2.schedulers import OperatorScheduler


# class IntSchema(BaseModel):
#     value: int


# class AddOneOperator(Operator):
#     input_schema = IntSchema
#     output_schema = IntSchema

#     def run(self, inp: IntSchema, **kwargs) -> Dict:
#         return {"value": inp.value + 1}


# class AddTwoOperator(Operator):
#     input_schema = IntSchema
#     output_schema = IntSchema

#     def run(self, inp: IntSchema, **kwargs) -> Dict:
#         return {"value": inp.value + 2}


# AddThreePipeline = Pipeline(
#     ops=[AddOneOperator(), AddTwoOperator()],
#     router=LinearRouter(end_route=2),
#     schedulers=[OperatorScheduler()],
# )


# def test_run_simple_pipeline():
#     pipeline_input = IntSchema(value=5)
#     pipeline_output = AddThreePipeline(pipeline_input)

#     assert pipeline_output.value == 8
    
    


# from tests.deepsparse.v2.dependency_injector.foo import Foo
# import time

# from dependency_injector import inject, containers, providers

# class Timer:
#     def __init__(self):
#         self.start_times = {}
#         self.measurements = {}
        
#     def start(id: str):
#         self.start_times[id] = time.time()
    
#     def stop(id: str):
#         self.measurements = time.time() - self.start_times[id]

# class TimerDependency:
#     def __init__(self, value):
#         self.timer = Timer()


# # Set up the dependency container
# class MyContainer(containers.DeclarativeContainer):
#     my_dependency = providers.Singleton(TimerDependency, value="Injected Value")

# # Resolve dependencies and create an instance of Foo
# container = MyContainer()
# foo = Foo()

# # Manually inject dependencies into the instance
# container.my_dependency.override("Manually Injected Value")
# breakpoint()

# # Call the method with injected dependencies
# foo.bar()


import logging
import sqlite3
from typing import Dict
from dependency_injector import containers, providers



class BaseService:

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}",
        )


class UserService(BaseService):

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db
        super().__init__()

    def get_user(self, email: str) -> Dict[str, str]:
        self.logger.debug("User %s has been found in database", email)
        return {"email": email, "password_hash": "..."}
    
    
class Container(containers.DeclarativeContainer):

    config = providers.Configuration(ini_files=["config.ini"])
    database_client = providers.Singleton(
        sqlite3.connect,
        config.database.dsn,
    )

    # Services

    user_service = providers.Factory(
        UserService,
        db=database_client,
    )


import sys

from dependency_injector.wiring import Provide, inject
# from .containers import Container


@inject
def main(
        # email: str,
        # password: str,
        # photo: str,
        user_service: UserService = Provide[Container.user_service],
) -> None:
    breakpoint()
    user = user_service.get_user(email)


def test():
    # if __name__ == "__main__":
    #     container = Container()
    #     container.init_resources()
    #     container.wire(modules=[__name__])
        
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])
    main()