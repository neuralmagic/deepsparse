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

To make sure, that the Custom Logger is compatible with our standards, it is required that it inherits from the `BaseLogger` class.

```python
from deepsparse.logging import BaseLogger, MetricsCategories

from typing import Any, Optional

class CustomLogger(BaseLogger):
    """
    The user has almost unlimited liberty to define any methods and attributes
    of the Custom Logger. 
    
    However, it is required that:
    1. the Custom Logger inherits from the BaseLogger class
    2. it implements a `log` method with the appropriate arguments
    """
    def log(self, identifier: str, value: Any, category: Optional[str]=None):
        """
        The main method to collect information from the pipeline
        
        :param identifier: The name of the item that is being logged.
            By default, in the simplest case, that would be a string in the form
            of "<pipeline_name>/<logging_target>"
            e.g. "image_classification/pipeline_inputs"
        :param value: The item that is logged along with the identifier
        :param category: The metric category that the log belongs to. 
            By default, we recommend sticking to our internal convention
            established in the MetricsCategories enum.
        """
```

Once a Custom Logger is implemented, it can be used in the DeepSparse Server, by specifying the appropriate
reference in the config yaml file:

```yaml
loggers:
    custom_logger: # the name of the logger can be chosen arbitrarily, but needs to be consistent with the rest of the config
        path: <path_to_the_script>/script.py:CustomLogger
        # optionally if the Custom Logger has an __init__() method,
        # the arguments can be specified here:
        arg1: some_argument_1
        arg2: some_argument_2
    # possibly other loggers
    ...
        
```

## Tutorial: Create Custom Logger That Sends Raw Image Data To S3
[@ROB todo] - demonstrate how to do this