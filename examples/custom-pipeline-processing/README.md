# Customized pre and post processing for Pipelines

Custom processors can be done via subclassing the Pipeline class you want to use
and then using dynamic import of the task in the server's config.yaml.

## Custom Pipeline

The first part of this is sub classing from whatever pipeline you want. You can see this
done in [custom_qa_pipeline.py](custom_qa_pipeline.py).

```python
...
class MyCustomQaPipeline(QuestionAnsweringPipeline):
    ...
```

The next step is to override the pre and post processing methods:

1. pre-processing: override `parse_inputs`
2. post-processing: override `process_engine_outputs`

## Dynamic import task

To use a task and pipeline from a custom python file you can use the "import:" prefix to specify
the file to load:

```yaml
task: "import:custom_qa_pipeline"
```

If you look at the [custom_qa_pipeline.py](custom_qa_pipeline.py) file, you'll notice that there's
a global variable named `TASK` and a call to `@Pipeline.register(TASK)` around the custom pipeline.

