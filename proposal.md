## UI for loggers + pipeline
```python
import Pipeline
pipeline = Pipeline.create(task="...", logger: Union[BaseLogger, str]) # str can be either a path or a yaml config
```
1. If we decide to pass a loaded logger (from `logger_from_config`), we can use it as is.
This would not be the typical path for the user- we will be missing the pipeline names in target identifiers tho.
2. The preferred way would be to pass a string, we can either load a logger from a yaml config or create a logger from a path. This enables us to call the `logger_from_config`
under the hood and inject the pipeline name into the target identifiers.

### Loading a preset data configs