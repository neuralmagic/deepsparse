## **Benchmark Performance**

DeepSparse include a convienent benchmarking script to test througput in a variety of scenarios.

As an example, let's take a look at throughput on a pruned-quantized version of BERT trained on SQuAD. DeepSparse
is integrated with SparseZoo, an open source repository of sparse models, so we can use SparseZoo stubs to 
download an ONNX file for testing.

On an AWS `c6i.4xlarge` instance (8 cores), DeepSparse achieves >300 items/second at batch 64.

```bash
deepsparse.benchmark zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none -i [64,128] -b 64 -nstreams 1 -s sync

>> Original Model Path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 323.7982
```

Run `deepsparse.benchmark --help` for full usage.

## **Deploy a Model**

DeepSparse offers two interfaces for deploying a model, a Python API called DeepSparse Pipelines and a REST API called DeepSparse Server. This gives you the flexibility to embed DeepSparse in an application and to deploy as a model service.

We will walk through an example of each.

### **DeepSparse Pipelines**

DeepSparse Pipelines wrap model inference with task-specific pre- and post-processing, such that you can pass raw inputs to
DeepSparse and recieve the post-processed output. For example, in the question answering domain, you can pass raw stings and recieve
the predicted answer, with DeepSparse handling the tokenization and answer extraction.

Let's take a look at an example in the question answering domain:

```python
from deepsparse import Pipeline

# downloads model from sparse zoo, compiles
qa_pipeline = Pipeline.create(task="question_answering", model_path="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none")

## run inference
prediction = qa_pipeline(question="What is my name?", context="My name is Snorlax")
print(prediction)
# >> score=19.847949981689453 answer='Snorlax' start=11 end=18
```

[Check out our documentation](https://docs.neuralmagic.com/use-cases/natural-language-processing/question-answering) to learn 
how to create a QA model trained on your data and deploy it with DeepSparse.

### **DeepSparse Server**

DeepSparse Server wraps Pipelines with the FastAPI web framework and Uvicorn
web server, making it easy to stand up a model service running DeepSparse.
Since the Server is a wrapper around Pipelines, you can send raw input
to the DeepSparse endpoint and recieve the post-processed answers.

DeepSparse Server is configured with YAML files that look like this:

```yaml
loggers:
  python:

endpoints:
  - task: question_answering
    model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none
```

