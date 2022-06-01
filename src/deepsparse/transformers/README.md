# Hugging Face Transformer Inference Pipelines


DeepSparse allows accelerated inference, serving, and benchmarking of sparsified [HuggingFace transformer](https://github.com/huggingface/transformers) models.  
This integration allows for leveraging the DeepSparse Engine to run the sparsified transformer inference with GPU-class performance directly on the CPU.

The DeepSparse Engine is taking advantage of sparsity within neural networks to 
reduce compute required as well as accelerate memory-bound workloads. The Engine is particularly effective when leveraging sparsification
methods such as [pruning](https://neuralmagic.com/blog/pruning-overview/) and [quantization](https://arxiv.org/abs/1609.07061). 
These techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics. 

This integration currently supports several fundamental NLP tasks:
- **Question Answering** - posing questions about a document.
- **Text Classification** - assigning a label or class to a piece of text (e.g Sentiment Analysis task). 
- **Token Classification** - attributing a label to each token in a sentence (e.g. Named Entity Recognition task).

We are actively working on adding more use cases, stay tuned!

## Getting Started


Before you start your adventure with the DeepSparse Engine, make sure that your machine is 
compatible with our [hardware requirements](https://docs.neuralmagic.com/deepsparse/source/hardware.html).

### Installation

```pip install deepsparse```

### Model Format
By default, to deploy the transformer using DeepSparse Engine it is required to supply the model in the ONNX format. 
This grants the Engine the flexibility to serve any model in a framework-agnostic environment. 

Below we describe two possibilities to obtain the required ONNX model.

#### Exporting the onnx file from the contents of a local directory
This pathway is relevant if you intend to deploy a model created using [SparseML](https://github.com/neuralmagic/sparseml) library. 
For more information, refer to the appropriate transformers integration documentation in SparseML.

The expected `model_path` in a transformers `Pipeline` should be a directory that includes the following files:
 - `model.onnx`
 - `tokenizer.json`
 - `config.json`

Onnx models can be exported using the `sparseml.transformers.export_onnx` tool:

```bash
sparseml.transformers.export_onnx --task question-answering --model_path model_path
```
This creates `model.onnx` file, in the parent directory of your `model_path`(e.g. `/trained_model/model.onnx`)

####  Directly using the SparseZoo stub
Alternatively, you can skip the process of onnx model export by downloading all the required model data directly from Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com/).
SparseZoo stubs which can be copied from each model page can be passed directly to a `Pipeline` to download and run
the sparsified ONNX model with its corresponding configs.


## Deployment

### Python API
Python API is the default interface for running inference with the DeepSparse Engine.

Once a model is obtained, either through `SparseML` training or directly from `SparseZoo`,
`deepsparse.Pipeline` can be used to easily facilitate end to end inference and deployment
of the sparsified transformers model.

If no model is specified to the `Pipeline` for a given task, the `Pipeline` will automatically
select a pruend and quantized model for the task from the `SparseZoo` that can be used for accelerated
inference. Note that other models in the SparseZoo will have different tradeoffs between speed, size,
and accuracy.


#### Question Answering Pipeline

[List of available SparseZoo Question Answering Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=question_answering)

```python
from deepsparse import Pipeline

qa_pipeline = Pipeline.create(task="question-answering")
inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")

>> {'score': 0.9947717785835266, 'start': 11, 'end': 18, 'answer': 'Snorlax'}
```

#### Sentiment Analysis Pipeline

[List of available SparseZoo Sentiment Analysis Models](
https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=sentiment_analysis)

```python
from deepsparse import Pipeline

# default model is a pruned + quantized text sentiment analysis model trained on sst2
sa_pipeline = Pipeline.create(task="sentiment-analysis")

inference = sa_pipeline("Snorlax loves my Tesla!")

>> [{'label': 'LABEL_1', 'score': 0.9884248375892639}]  # positive sentiment

inference = tc_pipeline("Snorlax hates pineapple pizza!")

>> [{'label': 'LABEL_0', 'score': 0.9981569051742554}]  # negative sentiment
```

#### Text Classification Pipeline

[List of available SparseZoo Text Classification Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=text_classification)

```python
from deepsparse import Pipeline

# using a pruned + quantized DistilBERT model from SparseZoo trained on QQP
tc_pipeline = Pipeline.create(
   task="text-classification",
   model_path="",
)

# inference of duplicate question pair
inference = tc_pipeline(
   sequences=[
      [
         "Which is the best gaming laptop under 40k?",
         "Which is the best gaming laptop under 40,000 rs?",
      ]
   ]
)

>> TextClassificationOutput(labels=['duplicate'], scores=[0.9947025775909424])
```

#### Token Classification Pipeline

[List of available SparseZoo Token Classification Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=token_classification)

```python
from deepsparse import Pipeline

# default model is a pruned + quantized NER model trained on the CoNLL dataset
tc_pipeline = Pipeline.create(task="token-classification")
inference = tc_pipeline("Drive from California to Texas!")

>> [{'entity': 'LABEL_0','word': 'drive', ...}, 
    {'entity': 'LABEL_0','word': 'from', ...}, 
    {'entity': 'LABEL_5','word': 'california', ...}, 
    {'entity': 'LABEL_0','word': 'to', ...}, 
    {'entity': 'LABEL_5','word': 'texas', ...}, 
    {'entity': 'LABEL_0','word': '!', ...}]
```

### DeepSparse Server
As an alternative to Python API, the DeepSparse inference server allows you to serve ONNX models and pipelines in HTTP.
Configs for the server support the same arguments as the above pipelines and setting the task and models to any of the
above transformers tasks and models will enable easy deployment.

For a full example of deploying sparse transformer models with the DeepSparse server, see the
[documentation](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server).

### Benchmarking
The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs. Want to find out how fast our sparse Hugging Face ONNX models perform inference? 
You can quickly do benchmarking tests on your own with a single CLI command!

You only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:

```bash
deepsparse.benchmark zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni

>> Original Model Path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni
>> Batch Size: 1
>> Scenario: multistream
>> Throughput (items/sec): 76.3484
>> Latency Mean (ms/batch): 157.1049
>> Latency Median (ms/batch): 157.0088
>> Latency Std (ms/batch): 1.4860
>> Iterations: 768
```

To learn more about benchmarking, refer to the appropriate documentation.
Also, check out our [Benchmarking tutorial](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)!

## Tutorials:
For a deeper dive into using transformers within the Neural Magic ecosystem, refer to the detailed tutorials on our [website](https://neuralmagic.com/):
- [Token Classification: Named Entity Recognition](https://neuralmagic.com/use-cases/sparse-named-entity-recognition/)
- [Text Classification: Multi-Class](https://neuralmagic.com/use-cases/sparse-multi-class-text-classification/)
- [Text Classification: Binary](https://neuralmagic.com/use-cases/sparse-binary-text-classification/)
- [Text Classification: Sentiment Analysis](https://neuralmagic.com/use-cases/sparse-sentiment-analysis/)
- [Question Answering](https://neuralmagic.com/use-cases/sparse-question-answering/)

## Support
For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues).
