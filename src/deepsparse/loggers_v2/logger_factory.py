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


import logging
import os
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict


class LoggerType(Enum):
    STREAM = logging.StreamHandler
    FILE = logging.FileHandler
    ROTATING = RotatingFileHandler


def create_file_if_not_exists(filename):
    if not os.path.exists(filename):
        open(filename, "a").close()

from abc import ABC
    
class RootLogger(ABC):
    # DEFAULT_LOGGER = {
    #     "python": "import logging"
    #     "promoetheus",
    # }
    
    def __init__(self, log_type_args: Any, logger_config: Dict, target: Dict):
        self.log_type_args = log_type_args
        self.logger_config = logger_config
        self.target = target
        
    def create(self):
        ...
        


    def log(self):
        ...
    
class SystemLogger(RootLogger):
    """
    Create Python level logging with handles
    """
    def __init__(self, log_type_args: str, logger_config: Dict, target: Dict):
        super().__init__(
            log_type_args=log_type_args, # for create compatibility with super
            logger_config=logger_config,
            target=target,
        )
        self.logger = logging.getLogger()  # Root loggger
        self.create()
        
        
    def create(self):
        system_logger_config = self.logger_config.get(self.log_type_args)
        self.logger.setLevel(system_logger_config.pop("level", "info"))

        handler = system_logger_config.pop("handler")
        if handler is not None:
            for handler_type, handler_config in handler.items():
                level = handler_config.pop("level", "INFO")
                handler = self.create_handler(handler_type, handler_config)
                handler.setLevel(level)
                self.logger.addHandler(handler)
        return self.logger

    def create_handler(self, handler_type, handler_config):
        logger_class = LoggerType[handler_type.upper()].value

        # Set handler level
        handler_level = handler_config.pop("level", logging.INFO)

        if handler_type == "stream":
            handler = logger_class()
        elif handler_type == "file":
            filename = handler_config.get("filename", "")
            create_file_if_not_exists(filename)
            handler = logger_class(filename=filename)
        elif handler_type == "rotating":
            filename = handler_config.get("filename", "")
            create_file_if_not_exists(filename)

            handler = logger_class(
                filename=filename,
                maxBytes=handler_config.get("max_bytes", 0),
                backupCount=handler_config.get("backup_count", 0),
            )
        else:
            raise ValueError(f"Unsupported logger type: {handler_type}")

        handler.setLevel(handler_level)

        # Set handler formatter
        formatter = logging.Formatter(handler_config.get("formatter", ""))
        handler.setFormatter(formatter)

        return handler
    
    def log(self, value, level: str = "info", **kwargs):
        do_log = getattr(self.logger, level)
        do_log(value, kwargs)
        
        
        
    # def create(self):
        
    #     loggers = {}
    #     for logger_id, targets in self.log_type_args.items():
    #         # find the logger from the logger id
    #         logger= self.logger_config.get(logger_id)
    #         logger_module = logger.get("use")
            
    #         if logger_module == "python":
    #             # use python module, skip target look up
    #             ...
    #         elif uses == "prometheus":
    #             # crate prometheus
    #             ...
    #         else:
    #             # custom
    #             ...
            
            
    #         # iterate thru the targets and find the target_args
            
            
            
    #         # loggers[logger_id] = (self.logger_config.get(logger_id), targets)
    #         # logger = self.logger_config.get(logger_id)
            
    #     for key, init_args in loggers:
    #         uses = logger.get("uses")
    #         if uses == "python":
    #             # create python
    #             ...
    #         elif uses == "prometheus":
    #             # crate prometheus
    #             ...
    #         else:
    #             # custom logger
    #             ...
                
                
                
            
            
            
            
        

        
            
        
            
            
            
        
        ...
        
    
        

class PerformanceLogger(RootLogger):
    ...
class MetricLogger(RootLogger):
    ...
    
    

ROOT_LOGGERS = {
    "system": SystemLogger,
    "performance": PerformanceLogger,
    "metric": MetricLogger,
}
 
def logger_factory(config: Dict) -> Dict[str, RootLogger]: 
    logger_config, target =config.get("logger"), config.get("target")
    
    loggers = {}
    for log_type, logger in ROOT_LOGGERS.items():
        log_type_args = config.get(log_type)
        if log_type_args is not None:
            loggers[log_type] = logger(
                log_type_args=log_type_args, 
                logger_config=logger_config,
                target = target,
            )
    return loggers
    

# class LoggerFactory:
    
   
    
#     def __init__(
#         self,
#         config: Dict,
#     ):
#         self.config = config
#         self.loggers = self.create()

#     def create(self):
#         loggers = {}
#         for log_type, logger in self.ROOT_LOGGERS.items():
#             init_args = self.config.get(log_type)
#             if init_args:
#                 loggers[log_type] = logger(init_args)
#         return loggers
