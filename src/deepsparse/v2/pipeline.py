# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List

from pydantic import BaseModel, Field, PrivateAttr

from deepsparse.v2.operators import Operator
from deepsparse.v2.routers import Router
from deepsparse.v2.schedulers import OperatorScheduler, SchedulerGroup


__all__ = ["Pipeline"]


class Pipeline(BaseModel):
    """
    Pipeline accepts a series of operators, schedulers, and a router which define
    an end to end ML transformation.

    Calling a pipeline runs these transformations
    """

    stages: List[Operator] = Field(
        required=True,
        description="In-order list of operators that make up this pipeline",
    )
    router: Router = Field(
        default_factor=Router,
        description="Router object to determine order and run the stages. "
        "Defaults to the base Router object",
    )
    schedulers: List[OperatorScheduler] = Field(
        default_factor=lambda: [OperatorScheduler()],
        description="List of schedulers to run operators in order of priority",
    )

    _scheduler_group: SchedulerGroup = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.validate()

        # SchedulerGroup handles running all schedulers in order of priority
        self._scheduler_group = SchedulerGroup(self.schedulers)

    def __call__(self, *args, return_context: bool = False, **kwargs):
        """
        :param return_context: if True, retrns tuple of the pipelien output
            and entire context. Default False
        :return: output of the pipeline stages ran with the router for the given input
        """
        if len(args) > 1:
            raise ValueError(
                "Only 1 in-line argument may be supplied to Pipeline which "
                f"must be a Schema, found: {len(args)}"
            )
        if args and kwargs:
            raise ValueError(
                "Pipeline can only run either a single in-line argument schema or a "
                f"series of kwargs, found {len(args)} args and {len(kwargs)} kwargs"
            )

        pipeline_input = args[0] or kwargs
        pipeline_output, context = self.router.run(
            inp=pipeline_input,
            operators=self.stages,
            scheduler=self._scheduler_group,
        )

        if return_context:
            return pipeline_output, context

        return pipeline_output

    def validate(self):
        router_validation = self.router.validate(self.stages)

        if router_validation is False:
            # default error message
            stage_types = [type(stage) for stage in self.stages]
            raise ValueError(
                f"Invalid Router: {type(self.router)} for stages: {stage_types}"
            )
        elif isinstance(router_validation, str):
            raise ValueError(f"Invalid Router for stages: {router_validation}")
