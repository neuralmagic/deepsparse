# Data Logging Functions

This page provides an overview of the data logging capabilities in DeepSparse Logging, including:
- List of predefined built-in functions that can be applied at each stage
- User guide for creating custom functions that can be applied at each stage

As a reminder, DeepSparse Pipelines have 4 `targets` for logging.
- `pipeline-inputs`: this stage is the raw inputs passed to the pipeline by the user
- `engine-inputs`: this stage is the tensors passed to the model (the output of pre-processing)
- `engine-outputs`: this stage is the raw output from the model
- `pipeline-outputs`: this stage is the post-processed data returned to the user

At each `target`, a list of functions can be applied in the form:

```yaml
target:
  mapping:
    # list of functions
    - func:       [REQUIRED STR: name of the function] << builtins/function_name OR path/to/custom.py:function_name
      frequency:  [OPTIONAL STR: how often to log (default = 10000, logs for 1/10000 inferences)]
      target:     [OPTIONAL STR: name of the logger to log to (default = all)]
    - func:
    ...
```

<details>
  <summary>Click for an example YAML file</summary>

```yaml
# logging-config.yaml

loggers:
  prometheus:
    port:5555
  s3logger:
    port:5556
    
data_logging:
  pipeline_inputs:
    mapping:
      - func: builtins/identify           # logs raw data to S3
        frequency: 10000
        target: s3logger
      - func: builtins/mean_pixel         # logs mean pixel value to prometheus
        frequency: 1000
        targets: prometheus

  engine_inputs:
    mapping:
      - func: path/to/custom.py:my_fn     # < logs a custom function of engine inputs
        # freqency: 1000                    < logs at default frequency of 1000
        # target: all                       < logs to default (all loggers)
  
  # engine_outputs      < nothing applied at this stage
  # pipeline_outputs    < nothing applied at this stage

```
</details>
  
## Built-in Functions

Built-in functions are predefined in DeepSparse. They are applied in the form `func: builtins/function_name`.
   
|Function Name  |Description                        |Compatible Loggers |Compatible Pipelines |Compatible Targets |
|---------------|-----------------------------------|-------------------|---------------------|-------------------|
|`mean_pixel_1` |Mean pixel value in first channel  |Prometheus         |Computer Vision      |`pipeline_inputs`  |
|`mean_pixel_2` |Mean pixel value in second channel |Prometheus         |Computer Vision      |`pipeline_inputs`  |
|`mean_pixel_3` |Mean pixel value in third channel  |Prometheus         |Computer Vision      |`pipeline_inputs`  |  
  
[@DAMIAN] - need help filling out this section

## Custom Functions

Users can defined custom functions in Python in a file, for example named `custom.py`. The custom functions are then specified in the YAML configuration file in the form `func: path/to/custom.py:function_name`.
  
An example Python file looks as follows:
  
```python
# custom.py

import numpy as np

def my_function(engine_inputs: np.array) -> float:
  return np.mean(engine_inputs)
```
The YAML file would look as follows:
``` yaml
# config.yaml
  
loggers:
  prometheus:
    port:5555
  
data_logging:
  pipeline_inputs:
    mapping:
      - func: path/to/custom.py:my_function
      - frequency: 1000
      - target: prometheus
```

In this way, user
  
[@DAMIAN] - can users pass arguments as well?
