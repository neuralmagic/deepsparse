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

from typing import Any

from deepsparse import Pipeline


__all__ = ["log_resource_utilization", "log_request_details"]


def log_resource_utilization(pipeline: Pipeline, **kwargs: Any):
    """
    Scope for 1.4:
    - CPU utilization overall
    - Memory available overall
    - Memory used overall (shall we continuously log this?
      this will be a constant value in time)
    - Number of core used by the pipeline
    """
    pass


def log_request_details(pipeline: Pipeline, **kwargs: Any):
    """
    Scope for 1.4:
    - Number of Successful Requests
    (binary events, 1 or 0 per invocation of an endpoint)
    - Batch size
    - Number of Inferences (
    number of successful inferences times the respective batch size)
    """
    pass
