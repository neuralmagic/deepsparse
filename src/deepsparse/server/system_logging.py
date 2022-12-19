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

from typing import Any, Dict, List

from deepsparse import Pipeline
from deepsparse.server.build_logger import extract_system_group_data
from deepsparse.server.config import ServerConfig


__all__ = ["SystemLoggingManager"]


class SystemLoggingManager:
    def __init__(self, system_group_names: List[str]):
        self.system_group_names = system_group_names

    @classmethod
    def from_server_config(cls, server_config: ServerConfig) -> "SystemLoggingManager":
        system_logging_config = server_config.system_logging
        system_group_names = extract_system_group_data(system_logging_config).keys()
        return cls(system_group_names)

    def log(self, pipeline: Pipeline, **kwargs: Dict[str, Any]):
        if not pipeline.logger:
            return
        for group_name in self.system_group_names:
            if group_name == "prediction_latency":
                self.log_prediction_latency(pipeline)

            elif group_name == "resource_utilization":
                self.log_resource_utilization(pipeline)

            elif group_name == "request_details":
                self.log_request_details(pipeline, **kwargs)

            elif group_name == "deployment_details":
                self.log_deployement_details(pipeline)

    def log_prediction_latency(self, pipeline: Pipeline):
        """
        Shall we control the prediction latency logging from here?
        During inference, we store the prediction values in the pipeline object
        and only pass it to the server logger if the system_logging_config allows for it
        """
        pass

    def log_resource_utilization(self, pipeline: Pipeline):
        pass

    def log_request_details(self, pipeline: Pipeline, **kwargs: Dict[str, Any]):
        pass

    def log_deployement_details(self, pipeline: Pipeline):
        pass
