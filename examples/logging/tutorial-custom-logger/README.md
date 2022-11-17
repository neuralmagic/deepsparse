# Custom Logger

This page explains how to create a Custom Logger for use with DeepSparse Logging.

### Why Might I Need A Custom Logger?
Prometheus has extremely strong support for 4 core [metric types](https://prometheus.io/docs/concepts/metric_types/) (Counter, Gauge, Histogram, and Summary), 
which cover many Monitoring use cases. However, Prometheus's model is not a great choice for logging high-dimensional raw data (such as a JPEG image or the tokenized
input to a Transformer). However, raw, unprojected data can sometimes be useful for various tasks, including:
- Debugging a pipeline
- Monitoring model accuracy with an offline data labeling
- Collecting a dataset for retraining the model with fresh data

Additionally, there are many MLOps projects/companies focused on Model Monitoring, which need the information from the inference 
pipelines to detect data drift. Custom Loggers enable you to connect DeepSparse deployment to these tools.

## How To Use The Custom Logger

Custom Loggers should inherit from the `ManagerLogger` class. 

[@DAMIAN please describe what to do]

```python

Class CustomLogger(ManagerLogger):
  def __init__
  ...

```

## Tutorial: Create Custom Logger That Sends Raw Image Data To S3
[@ROB todo] - demonstrate how to do this
