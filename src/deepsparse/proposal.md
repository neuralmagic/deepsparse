Let's imagine our example system logging config:
```yaml
system_logging:
  enable: true/false
  target_loggers:
    - python
    - prometheus
  inference_latency_group:
         enabled: true/false
         target_loggers:
           - python
           - prometheus
  resource_group:
         enabled: true/false
```

This would add additional two function loggers to our overall set of loggers. We would translate all the logging configs into a 
following `server_logger`:
```python
server_logger = AsyncLogger(logger = MultiLogger(
    loggers = [
        FunctionLogger_1(...),
        FunctionLogger_2(...),
        ...
        FunctionLogger_n(...) # FunctionLoggers from 1 to n are pertaining to the data logging config
        FunctionLogger_n_+_1(...) # Pertains to inference_latency logging (frequency = 1; target_identifier = "system/inference_latency"). Acts a "filter" for all the inference latency logs
        FunctionLogger_n_+_2(...) # Pertains to resource logging (frequency = 1; target_identifier = "system/resource_utilization"). Acts a "filter" for all the resource logs
        ... # possibly pertain to other system loggers
    ]))
```

I like that every function logger has a single responsibility of "detecting" and logging only one thing, whether it is 
data logging target or system data group. They can easily deal with `target_loggers`. However, there is one imperfection here: even if all the system logging is disabled, 
we would still compute the inference latency or recourse utilization. In the long run, when add more features that may be quite wasteful.

This is why I propose to create an object called (for now) `SystemLoggingManager` that would be responsible for managing the system logging.
The purpose is to avoid computing any data if it is not needed.

```python

class SystemLoggingManager:
    def __init__(self, server_logger: BaseLogger):
        # not use server_logger but use system_log_config
        self.system_groups_names: List[str] = self.get_system_groups_names(server_logger) # e.g. ["resource, inference_latency", ...]. Can be inferred from the target_identifiers

    def log_system_info(self, pipeline: Pipeline):
        # this function would also handle the frequency of the system logging
        # we do not want to compute anything if it is not needed
        for group_name in self.system_groups_names:
            if group_name == "resource":
                self.log_resource_utilization(pipeline)
            if group_name == "deployment_details":
                self.log_deployment_details(pipeline)
            ...
        
    def log_resource_utilization(self, pipeline):
        cpu_utilization, mem_utilization, total_mem = _compute_resource_utilization()
        
        # log all resource utilization data to the system logger
        pipeline.logger.log(identifier = "system/resource_utilization/cpu_utilization", category = "system", value = cpu_utilization)
        ...

    def log_deployment_details(self, pipeline):
        batch_size, ... = pipeline.engine.batch_size
        
        # log all resource utilization data to the system logger
        pipeline.logger.log(identifier = f"{pipeline.name}/system/deployment_details/batch_size", category = "system", value = batch_size)
        ...
        ...
```
The only required code change would be:
```python

server_logger = build_logger(server_config)
system_logging_manager = SystemLoggingManager(server_logger)

 def _predict(request: pipeline.input_schema):
        pipeline_outputs = pipeline(request)
        if pipeline.logger:
            system_logging_manager.log_system_info(pipeline)
        return pipeline_outputs
```

Request Details
    Batch Size
    Number of Successful Requests
    Number of Failed Requests
    Number of Inferences
    Number of Executions
Prediction Latency - complete in 1.3 -> enable/disable 
    Pre-process Time
    Engine Time
    Post-process Time
    Total Inference Time
Resource Utilization -> how often (secondary concern)
    CPU Utilization - how much % CPU is used by the main python thread (server)
    Memory Available
    Memory Used
Deployment Details -> YOLO
    Device ID, Type, Name, RAM, CPUs, etc
    Number of Cores Used
    Model Name, Model Version, Model Image

