# Rough notes from 1:1

# Sample Logging Config

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
# flake8: noqa
"""
LoggingConfig:
    system:  # python logging-like logs
        level: str

    performance:  # Timings, system (CPU usage, etc)
        frequency: float
        timings: bool = ...
        cpu: bool = ...
        # each performance log can be enabled or disabled by a bool in the config

    metrics:  # Data logs, etc
        # operator name format can either be name of operator as defined in graph router
        # or the operator name and a specific key in its output dictionary
        **operator_name: MetricLoggingConfig

"""

"""
# reuse or port: https://github.com/neuralmagic/deepsparse/blob/eb039469f341f6abf4b4a3e5d8da0d8ad885c162/src/deepsparse/loggers/config.py#L32
MetricLoggingConfig:
    function: str
    frequency: str
"""

"""
Example config

system:
    level: DEBUG
    
performance:
    frequency: 1.0
    timings: True
    cpu: False
    
metrics:
    process_input.prompt:
        function: identity
        frequency: 0.1  # log 10% of these
        
    prep_for_generation.logits:
        function: max
        frequency: 1.0
"""

"""
Example for pipeline:

Pipeline.create(**kwargs, logging: LoggingConfig=...)

inside the pipeline, we would instantiate the (async) logger from a config

if no logging config is provided, no logging is done (make a null logger if easier to implement)
"""

"""
UX

we want to reuse as much of the existing logger as possible
but we will to update the UX a bit to support the new target of
system, performance, metrics

for now let's target UX

LogType(Enum):
    system
    performance
    metrics

# logger entrypoints
logger.system.log
logger.performance.log
logger.metrics.log

# pipeline log entrypoint
pipeline.log(type: LogType, value: Any, tag: Optional[str]):
    # split on type and call into self.logger....
    
    
###
Integrating the logger

Metric Logging:
    * use middleware so after an operator runs, it will optionally log its
outputs as metric logs
    ideally this will it easy to flow from a config that defines target operators
    to get their values logged

Performance Logging:
    * update timing middleware to fire performance logs
    * splash in the optional CPU/utilization performance logs into the pipeline

System Logging:
    * individual operators should have access to the logger from `PipelineState`
so that they can log system logs to the appropriate level
"""
