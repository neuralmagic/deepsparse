Our identifiers and matching logic seems to get a bit unruly, mainly because of ad-hoc decision taken during the engineering works. Let's agree on some standards.


# The design of the logged identifier before it reaches the FunctionLogger

## Identifiers that are being logged by the pipeline:

The general template is the following:
```python
Identifier: f"{pipeline_identifier OR compound_pipeline_identifier}/{main_identifier OR compound_main_identifier}"
```
e.g.
```python
Identifier_1: "question_answering_pipeline/pipeline_outputs", Category_1: MetricCategories.DATA
Identifier_2: "pose_estimation_pipeline/object_detection_pipeline/pipeline_outputs/specific_embeddings", Category_2: MetricCategories.DATA # compound identifier + "/" + compound main identifier
Identifier_3 : "pose_estimation_pipeline/object_detection_pipeline/pre_processing", Category_3: MetricCategories.SYSTEM
```
The general rule, that every component of an identifier, whether it is the pipeline name or main identifier, regardless of the
the complexity of the identifier is always connected by "/".

Note: some of the system metrics may have values computed externally but logged internally by the pipeline.
```python
value = external_function()
identifier = "some_value"
pipeline.log(identifier = identifier, value = value, category = MetricCategories.SYSTEM) 
# -> will be logged as f"{pipeline_identifier OR compound_pipeline_identifier}/some_value"
```

## Identifiers that are being logged externally (system_logging)
```python
Identifier: f"{main_identifier OR compound_main_identifier}"
```
e.g.

```python
Identifier_1: "cpu_utilization", Category_1: MetricCategories.DATA
Identifier_2: "cpu_utilization/core_1", Category_2: MetricCategories.DATA
```
# Matching Logic for the Function Logger

## For Data Logging
In the `build_logger` function, we create the target_identifier by concatenating the pipeline_identifier and the main_identifier.
```python
target_identifier = f"{endpoint_name}/{target_name}" # if not regex
target_identifier = target # if regex
```
Since target_identifier creation follows the same logic as the identifier creation for the pipeline, the matching is straightforward.

## For System Logging
I think we should follow a similar as the data logging, and aim for 1:1 matching between the identifier and the target_identifier.
```yaml
system_logging:
    prediction_latency:
      ...
    ```python
```
should result in 
`target_identifier = "prediction_latency"` 
`identifier = prediction_latency`
so the matching is straightforward.

Effectively, we should stop prepending the "category:" string.

If we employ regex
```yaml
system_logging:
    re:something_something:
      ...
    ```python
```

We act analogously to data logging.

Such an approach should simplify much of the logic that we have right now.
So all in all our matching logic will look like this (`check_identifier_match` function):
```python
 if template[:3] == "re:": # handling regex cases for both SYSTEM AND DATA categories
    pattern = template[3:]
    return re.match(pattern, identifier) is not None, None
if template == identifier: # handling "clean" matches for both SYSTEM AND DATA categories
    return True, None
if template.startswith(identifier): # handling the remaining matches for both SYSTEM AND DATA categories
    remainder = template.replace(identifier, "")
    # Note: this will require us to avoid calling possibly_extract_value() on the remainder if category == MetricCategories.SYSTEM.
    # This is because the remainder of such a log should never contain any components that should be used for indexing/slicing/accessing the value.
    return True, remainder if remainder.startswith("[") else remainder[1:]

return False, None
```

# On the output from the FunctionLogger
- if logging belongs to the DATA category and the remainder has slicing/indexing/access information, 
  remove it e.g. "some_value[0, 1:2]" -> "some_value". Then concatenate it with the identifier using ".": "identifier.some_value"
- when the function is applied to the value, concatenate it with the identifier using "_": "identifier.some_value_{function_name}". 
  If the identifier belongs to the SYSTEM category (function is identity function), skip this step.










