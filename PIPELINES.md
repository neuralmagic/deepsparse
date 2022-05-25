# Working with DeepSparse Pipelines

## Introduction

DeepSparse Pipelines provide a simple interface for end-to-end ML inference that wraps
the [DeepSparse Engine](https://github.com/neuralmagic/deepsparse) with task specific
processing.  Pipelines are created for a given task and include support for transforming
raw inputs into fully processed predictions with sparse acceleration.

Inputs and outputs can be given as [pydantic](https://pydantic-docs.helpmanual.io/)
schemas or may be parsed into them. This provides out of the box type checking and
validation for Pipelines.

Pipelines are created through the `Pipeline.create(...)` method.  `create` requires a
`task` name, an optional `model_path`, as well as other task specific key word arguments
for deployment.  Full documentation can be found in
[`pipelines.py`](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/pipeline.py)
and task specific docs.

While a default model for each task will be downloaded when `model_path` is not specified,
however `model_path` may be a [SparseZoo](https://github.com/neuralmagic/deepsparse)
model stub, path to a local ONNX file, or deployment directory with an ONNX file.

### Example Usage

```python
from deepsparse import Pipeline

# create a QuestionAnsweringPipeline with a default sparse QA model
qa_pipeline = Pipeline.create(task="question-answering")

# run sample inference
qa_pipeline(question="who is mark?", context="mark is batman")
```

## Supported Tasks

Development of new Pipelines for tasks is always ongoing. Currently supported tasks include:

| Domain |         Task         | Documentation |
|--------|:--------------------:|:-------------:|
| NLP    | text-classification  |      TBA      |
| NLP    | token-classification |      TBA      |
| NLP    |  question-answering  |      TBA      |
| CV     | image-classification |      TBA      |
| CV     |   object-detection   |      TBA      |


## Deployment

DeepSparse Pipelines are tightly integrated with the DeepSparse model server and which
can also be deployed to Amazon SageMaker at scale.  Any task supported by `Pipeline.create`
can easily be added to the config of these deployments.

* [deepsparse.server](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/server/README.md)
* [deepsparse-SageMaker](https://github.com/neuralmagic/deepsparse/blob/main/examples/aws-sagemaker/README.md)

## Custom Tasks

Custom tasks may be defined by inheriting from the base Pipeline class in 
[`pipelines.py`](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/pipeline.py)
and implementing the `abstractmethod` stubs.  This includes defining input and output
pydantic schemas and defining pre and post processing flows.

Tasks may also be registered to `Pipeline.create` by decorating the implemented class
with `Pipeline.register`. This enables easy instantiation of the new pipeline in existing
scripts and flows by refering to the new task name.

#### `Pipeline.register` Example
```python
from deepsparse import Pipeline

# register and define custom pipeline
@Pipeline.register(task="my-custom-task")
class MyCustomPipeline(Pipeline):
    pass

...

# create MyCustomPipeline instance using Pipeline.create
custom_pipeline = Pipeline.create(task="my-custom-task", model_path="...")
```

## Support

For Neural Magic Support, sign up or log in to our
[Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ).
Bugs, feature requests, or additional questions can also be posted to our
[GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues).
