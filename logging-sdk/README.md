# DeepSparse Logging

DeepSparse Logging provides operational teams with access to the telemetry needed to monitor a deployment. 

For users seeking to put ML into production, these data are the raw materials that underpin the monitoring processes needed to create a system with consistently fresh and accurate predictions.

<p align="center">
     <img src="continual-learning.png"
          alt="Continual Learning Diagram"
          width="50%"
     />
</p>

There are many types of downstream monitoring tasks that ML teams may want to perform. Some are easier to do (such as looking at system performance) and some are harder (looking at model accuracy requires manually labeling some data afterwards). Examples include:
- **System performance:** what is the latency/throughput of a query?
- **Data quality:** is there an error in the model pipeline?
- **Data distribution shift:** are the inputs and outputs from the system expected?
- **Model accuracy:** what is the accuracy of the predictions vs human-labeled?

DeepSparse Logging is designed to provide maximum flexibility for users to extract whatever data is needed from a production inference pipeline into the logging system of their choice. 

## Metrics 
DeepSparse Logging provides access to two types of metrics:
- **System Logging Metrics** give operations teams access to granual performance metrics, diagnosing and isolating deployment system health. Examples include CPU utilization and query latency.
     - [List of System Logging Metrics](https://github.com/neuralmagic/deepsparse/blob/rs-logging-sdk/logging-sdk/system-logging-metrics.md)

- **Data Logging Metrics** give ML teams access to data at each stage of an ML pipeline, supporting downsteam tasks like measuring accuracy and data drift. Examples include raw inputs and projections thereof such as mean pixel value.
     - [List of Built-in Data Logging Functions](https://github.com/neuralmagic/deepsparse/blob/rs-logging-sdk/logging-sdk/data-logging-functions.md#built-in-functions)
     - [Custom Data Logging Functions](https://github.com/neuralmagic/deepsparse/blob/rs-logging-sdk/logging-sdk/data-logging-functions.md#custom-functions) 

## Configuration
DeepSparse Logging is configured via YAML files.

<details>
    <summary><b>System Logging</b></summary>
    </br>

System Logging is *enabled* by default and all metrics are pre-defined. System Logging can be disabled 
globally or at the Group level by adding the key-value pairs with `on` or `off` status.

The following format is used:

```yaml
# system_logging: on/off            # [OPTIONAL] flag to turn all system logging on/off; note: if omitted, defaults to on

system_logging:
    deployment_details: on/off      # [OPTIONAL] flag to turn off deployment details group; if omitted, defaults to on
    request_details: on/off         # [OPTIONAL] flag to turn off request details group; if omitted, defaults to on
    prediction_latency: on/off      # [OPTIONAL] flag to turn off prediction latency group; if omitted, defaults to on
    dynamic_batch_latency: on/off   # [OPTIONAL] flag to turn off dynamic batch latency group; if omitted, defaults to on
    resource_utilization: on/off    # [OPTIONAL] flag to turn off resource utilization group; if omitted, defaults to on     
```

A tangible example YAML snippit is below:

```yaml
# system_logging: off                << note: optional flag to turn off everything

system_logging:
    deployment_details: off
    request_details: off
    prediction_latency: on 
    dynamic_batch_latency: off
    # resource_utilization: on      << note: omitted groups are turned on by default
```
In this example, system logging is turned on globally. The Deployment Details, Request Details, and Dynamic Batch Latency groups are turned off while Prediction Latency and Resource Utilization groups are turned on.

</details>

> :warning: System Logging is ***enabled*** by default

<details>
    <summary><b>Data Logging</b></summary>
    </br>
        
Data Logging is *disabled* by default. A YAML configuration file is used to specify which data or functions thereof to log.

There are 4 `stages` in the inference pipeline where Data Logging can occur:

|Stage            |Pipeline Inputs      |Engine Inputs  |Engine Outputs     |Pipeline Outputs   |
|--------------   |---------------------|---------------|-------------------|-------------------|
|**Description**  |Inputs passed by user|Preprocessed tensors passed to model|Outputs from model (logits)|Postprocessed output returned to user|
|`stage`      |`pipeline_inputs`    |`engine_inputs`|`engine_outputs`   |`pipeline_outputs` |
    
The following format is used to apply a list of [built-in](https://github.com/neuralmagic/deepsparse/blob/rs-logging-sdk/logging-sdk/data-logging-functions.md#built-in-functions) and/or [custom functions](https://github.com/neuralmagic/deepsparse/blob/rs-logging-sdk/logging-sdk/data-logging-functions.md#custom-functions) to a Pipeline `stage`:
 
```yaml     
stage:                              # options: pipeline_inputs, engine_inputs, engine_outputs, pipeline_outputs
  mapping:
    # first function
    - func: builtins/identity       # [REQUIRED STR] function identifier  (built-in or path to custom)
      frequency: 1000               # [OPTIONAL INT] logging frequency    (default: 1000 - logs once per 1000 predictions)
      target: all                   # [OPTIONAL STR] logger               (default: all)
    # second function
    - func: path/to/custom.py:my_fn  
      frequency: 10000
      target: prometheus
  
  # ... list of as many functions as desired
```

A tangible example YAML snippit is below:

```yaml
pipeline_inputs:
  mapping:
    - func: builtins/identity                   # built-in function (logs raw data)
      frequency: 100                            # logs raw data once per 100 predictions
      target: prometheus                        # only logs to prometheus
    - func: /path/to/logging_fns.py:my_fn       # custom function
      # frequency:                              # not specified, defaults to once per 1000 predictions
      # target:                                 # not specified, defaults to all loggers

engine_inputs:
  mapping:
    - func: builtins/channel-mean               # pre-defined function (logs per channel mean pixel)
      frequency: 10                             # logs channel-mean once per 10 predictions
      # target:                                 # not specified, defaults to all loggers

# engine_outputs:                             # not specified, so not logged
# pipeline_outputs:                           # not specified, so not logged
```
This configuration does the following at each stage of the Pipeline:
- **Pipeline Inputs**: Raw data (from the `identity` function) is logged to Prometheus once every 100 predictions and a custom function called `my_fn` is applied once every 1000 predictions and is logged to all loggers.
- **Engine Inputs**: The `channel-mean` function is applied once per 10 predictions and is logged to all loggers.
- **Engine Outputs**: No logging occurs at this stage
- **Pipeline Outputs**: No logging occurs at this stage

</details>

> :warning: Data Logging is ***disabled*** by default

## Loggers
DeepSparse Logging provides users with ability to log to Prometheus out-of-the-box as well as the ability to add custom loggers.

### Built-In Loggers

[TBD] - what should we say about high dimensional (images/vectors) vs metrics

[@DAMIAN] - need your help here

### Custom Loggers

[@DAMIAN] - need your help here

There is a [tutorial creating a custom logger to an S3 Bucket](https://github.com/neuralmagic/deepsparse/tree/rs-logging-sdk/logging-sdk/tutorial-custom-logger) as an example.

## Usage 

Both the `Server` and `Pipeline` interfaces can run with logging.

<details> 
    <summary><b>Server Usage</b></summary>
    </br>

The startup command (`deepsparse.server`), accepts an optional YAML configuration file (which contains both logging-specific and general configuration details) via the `--config` argument. For example:

```bash
deepsparse.server --config config.yaml
```

The logging is configured as described above. System Logging is defined globally at the `Server` level while Data Logging is defined at the `Endpoint` level. In the example below, we create a `Server` with two `Endpoints` (one with a dense BERT model and one with a sparse BERT model). We can see that System Logging is defined globally and Data Logging is defined per `Endpoint`.

```yaml
# config.yaml

loggers:
     -prometheus
     -custom_logger

system_logging:
     deployment_details: off
     request_details: off
     prediction_latency: on 
     dynamic_batch_latency: off
     # resource_utilization:                      < not specified, so enabled by default

# system_logging: off                             < optional flag to turn off all system logging

endpoints:
     - task: question_answering
       route: /dense/predict
       model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
       batch_size: 1
       data_logging:
          - target: pipeline_inputs
               mappings:
                - func: builtins:sequence_length
                  frequency: 500
                  target: prometheus
                - func: path/to/custom/metrics.py:fn2
                  # frequency:                    < not specified, so logs at default rate of 1/1000 inferences
                  # target:                       < not specified, so logs to all
             - target: engine_inputs
                mappings:
                - func: buildins:identity
                  frequency: 1000
                  target: custom_logger
             # - target: engine_outputs           < not specified, so nothing logged at this target
             # - target: pipeline_outputs         < not specified, so nothing logged at this target
             
     - task: question_answering
       route: /sparse/predict
       model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none
       batch_size: 1
       # data_logging:                            < not specified, so no data logging for this endpoint
```

</details>

<details> 
    <summary><b>Pipeline Usage</b></summary>
    </br>

The `ManagerLogger` class will handles logging for a `Pipeline`. It is passed as the `logger` argument to a `Pipeline`. `ManagerLogger` is initialized with the `config` argument, which is a path to a local logging configuration file in the format described above. 

If no `ManagerLogger` is passed to a `Pipeline`, then logging is disabled.

[@DAMIAN] - how does it work with a custom-pipeline + multi-model engine?

For example, with the QA pipeline:

```python
from deepsparse import Pipeline

# SparseZoo model stub or path to ONNX file
model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

# logger object referencing the local logging config file
logger = ManagerLogger(config="logging-config.yaml")

# pipeline instantiated with the config file
pipeline = Pipeline.create(
    task="question-answering",
    model_path=model_path,
    logger=logger
)

my_name = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```

```yaml
# logging-config.yaml

loggers:
     -prometheus
     -custom_logger

system_logging:
     deployment_details: off
     request_details: off
     prediction_latency: on 
     dynamic_batch_latency: off
     # resource_utilization:               < not specified, so enabled by default

# system_logging: on                       < optional flag to turn off all system logging; OPTIONS: [ON/OFF]

data_logging:
  - target: pipeline_inputs
       mappings:
        - func: builtins:sequence_length
          frequency: 500
          target: prometheus
        - func: path/to/custom/metrics.py:fn2
          # frequency:                    < not specified, so logs at default rate of 1/1000 inferences
          # target:                       < not specified, so logs to all
     - target: engine_inputs
        mappings:
        - func: buildins:identity
          frequency: 1000
          target: custom_logger
     # - target: engine_outputs           < not specified, so nothing logged at this target
     # - target: pipeline_outputs         < not specified, so nothing logged at this target
```
</details>

## Tutorials
There are serveral examples integrating Prometheus and DeepSparse logging available:
- [Server with Prometheus / Grafana](https://github.com/neuralmagic/deepsparse/tree/rs-logging-sdk/logging-sdk/tutorial-server-prometheus)
- [Pipeline with Prometheus / Grafana](https://github.com/neuralmagic/deepsparse/tree/rs-logging-sdk/logging-sdk/tutorial-pipeline-prometheus)
- [Monitoring while running Kubernetes](https://github.com/neuralmagic/deepsparse/tree/rs-logging-sdk/logging-sdk/tutorial-kubernetes-prometheus)
- [Custom Logger (S3 Bucket)](https://github.com/neuralmagic/deepsparse/tree/rs-logging-sdk/logging-sdk/tutorial-custom-logger)

##  Data Logging Functions

**TO BE UPDATED** ONCE WE HAVE LIST FOR 1.3
  
### Built-in Functions

Built-in functions are predefined in DeepSparse. 
   
|Function Name  |Description                        |Compatible Loggers |Compatible Pipelines |Compatible Targets |
|---------------|-----------------------------------|-------------------|---------------------|-------------------|
|`mean_pixel_1` |Mean pixel value in first channel  |Prometheus         |Computer Vision      |`pipeline_inputs`  |
|`mean_pixel_2` |Mean pixel value in second channel |Prometheus         |Computer Vision      |`pipeline_inputs`  |
|`mean_pixel_3` |Mean pixel value in third channel  |Prometheus         |Computer Vision      |`pipeline_inputs`  |  

They are applied in the form `func: builtins/function_name`. The YAML file would look as follows:
``` yaml
# config.yaml
  
loggers:
  prometheus:
    port:5555
  
data_logging:
  pipeline_inputs:
    mapping:
      - func: builtin:function
      - frequency: 1000
      - target: prometheus
``` 

### Custom Functions

Users can defined custom functions in Python in a file, for example `custom.py`, which should be stored in the filesystem of the application. 
  
The custom functions are then specified in the YAML configuration file in the form `func: path/to/custom.py:function_name`.
  
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

## System Logging Metrics

**TO BE UPDATED** ONCE WE HAVE THE LIST OF METRICS FOR 1.3

Below is the list and details details on the System Logging Metrics.

### Groups
The following metric groups are enabled by default:
|Group                |Description  |Identifier in YAML Config Files|
|---------------------|-------------|-------------------------------|
|Deployment Details   |xxx          |`deployment_details`           |
|Request Details      |xxx          |`request_details`              |
|Prediction Latency   |xxx          |`prediction_latency`           |
|Engine Batch Latency |xxx          |`batch_latency`                |
|Resource Utilizaiton |xxx          |`resource_utilization`         |

### Full List of Metrics
The following metrics are logged for each group:

|Group              |Metric           |Metric Name              |Description                              |Granularity    |Usage  |Frequency      |
|-------------------|---------------- |-------------------------|-----------------------------------------|---------------|-------|---------------|
|Prediction Latency |Total Time       |`sl_pl_total_time`       |End-to-end prediction time               |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Preprocess Time  |`sl_pl_preprocess_time`  |Time spent in pre-processing step        |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Queue Time       |`sl_pl_queue_time`       |Time spent in queue (waiting for engine) |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Engine Time      |`sl_pl_engine_time`      |Time spent in engine forward pass        |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Postprocess Time |`sl_pl_postprocess_time` |Time spent in post-processing step       |Per Pipeline   |All    |Per Prediction |

## DeepSparse Enterprise

Interested in running in production? [Try DeepSparse Enterprise for 90 days free](link/to/trial/page).
