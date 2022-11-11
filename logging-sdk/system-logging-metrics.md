# System Logging Metrics

**TO BE UPDATED** ONCE WE HAVE THE LIST OF METRICS FOR 1.3

This page includes details on the individual System Logging Metrics.

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
|Deployment Details |Model Name       |`sl_dd_model_name`       |Name of the model running                |Per Pipeline   |All    |1 hour         |
|Deployment Details |CPU Arch         |`sl_dd_cpu_arch`         |Architecture of the CPU                  |Per Server     |All    |1 hour         |
|Deployment Details |CPU Model        |`sl_dd_cpu_model`        |Model of the CPU                         |Per Server     |All    |1 hour         |
|Deployment Details |Cores Used       |`sl_pl_num_cores`        |Number of cores used by the engine       |Per Server     |All    |1 hour         |
|Prediction Latency |Total Time       |`sl_pl_total_time`       |End-to-end prediction time               |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Preprocess Time  |`sl_pl_preprocess_time`  |Time spent in pre-processing step        |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Queue Time       |`sl_pl_queue_time`       |Time spent in queue (waiting for engine) |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Engine Time      |`sl_pl_engine_time`      |Time spent in engine forward pass        |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Postprocess Time |`sl_pl_postprocess_time` |Time spent in post-processing step       |Per Pipeline   |All    |Per Prediction |



