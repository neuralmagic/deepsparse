*LAST UPDATED: 10/11/2023*

# **Sparse Finetuned LLMs with DeepSparse**

DeepSparse has support for performant inference of sparse large language models, starting with Mosaic's MPT.

In this research overview, we will discuss:
1. [Our Sparse Fineuning Research](#sparse-finetuning-research)
2. [How to try Text Generation with DeepSparse](#try-it-now)

## **Sparse Finetuning Research**

We show that MPT-7B can be pruned to ~60% sparsity with INT8 quantization (and 70% sparsity without quantization), with no accuracy drop, using a technique called **Sparse Finetuning**, where we prune the network during the finetuning process.

When running the pruned network with DeepSparse, we can accelerate inference by ~7x over the dense-FP32 baseline!

### **Sparse Finetuning on Grade-School Math (GSM)**

Training LLMs consist of two steps. First, the model is pre-trained on a very large corpus of text (typically >1T tokens). Then, the model is adapted for downstream use by continuing training with a much smaller high quality curated dataset. This second step is called finetuning.

Fine-tuning is useful for two main reasons:
1. It can teach the model *how to respond* to input (often called **instruction tuning**).
2. It can teach the model *new information* (often called **domain adaptation**).


An example of how domain adaptation is helpful is solving the [Grade-school math (GSM) dataset](https://huggingface.co/datasets/gsm8k). GSM is a set of grade school word problems and a notoriously difficult task for LLMs, as evidenced by the 0% zero-shot accuracy of MPT-7B. By fine-tuning with a very small set of ~7k training examples, however, we can boost the model's accuracy on the test set to 28.2%.

The key insight from our paper is that we can prune the network during the finetuning process. We apply [SparseGPT](https://arxiv.org/pdf/2301.00774.pdf) to prune the network after dense finetuning and retrain for 2 epochs with L2 distillation. The result is a 60% sparse-quantized model with no accuracy drop on GSM8k runs 7x faster than the dense baseline with DeepSparse!

<div align="center">
    <img src="https://github.com/neuralmagic/deepsparse/assets/3195154/8687401c-f479-4999-ba6b-e01c747dace9" width="60%"/>
</div>

- [See the paper on Arxiv]() << UPDATE >>

### **How Is This Useful For Real World Use?**

While GSM is a "toy" math dataset, it serves as an example of how LLMs can be adapted to solve tasks which the general pretrained model cannot. Given the treasure-troves of domain-specific data held by companies, we expect to see many production models fine-tuned to create more accurate models fit to business tasks. Using Neural Magic, you can deploy these fine-tuned models performantly on CPUs!

## Try It Now

Install the DeepSparse Nightly build (requires Linux):

```bash
pip install deepsparse-nightly[transformers]==1.6.0.20231007
```

The models generated in the paper are hosted on [SparseZoo](https://sparsezoo.neuralmagic.com/?ungrouped=true&sort=null&datasets=gsm8k&architectures=mpt) and [Hugging Face](https://huggingface.co/collections/neuralmagic/sparse-finetuning-mpt-65241d875b29204d6d42697d). 

### MPT-7B on GSM 

We can run inference on the models using DeepSparse's `TextGeneration` Pipeline:

```python
from deepsparse import TextGeneration

model = "zoo:mpt-7b-gsm8k_mpt_pretrain-pruned60_quantized"
pipeline = TextGeneration(model_path=model)

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

### >> Natalia sold 48/2 = <<48/2=24>>24 clips in May.
### >> Natalia sold 48 + 24 = <<48+24=72>>72 clips altogether in April and May.
### >> #### 72
```

It is also possible to run the models directly from Hugging Face by prepending `"hf:"` to a model id, such as:

```python
from deepsparse import TextGeneration

hf_model_id = "hf:neuralmagic/mpt-7b-gsm8k-pruned60-quant"
pipeline = TextGeneration(model=hf_model_id)

prompt = "Question: Marty has 100 centimeters of ribbon that he must cut into 4 equal parts. Each of the cut parts must be divided into 5 equal parts. How long will each final cut be?"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

### >> The 100-centimeter ribbon is cut into 4 equal parts in pieces of 100 / 4 = <<100/4=25>>25 cm
### >> From each piece of 25 cm, he gets 5 equal parts of 25 / 5 = <<25/5=5>>5 cm each.
### >> #### 5
```

> **Note:** DeepSparse uses ONNX graphs modified for KV-caching. We will publish specs to enable external users to create LLM ONNX graphs for DeepSparse over the next few weeks. ***At current, we suggest only using LLM ONNX graphs created by Neural Magic's team***


#### Other Resources
- [Check out all the MPT GSM models on SparseZoo](https://sparsezoo.neuralmagic.com/?datasets=gsm8k&ungrouped=true)
- [Try out the live demo on Hugging Face Spaces](https://huggingface.co/spaces/neuralmagic/sparse-mpt-7b-gsm8k) and view the [collection of paper, demos, and models](https://huggingface.co/collections/neuralmagic/sparse-finetuning-mpt-65241d875b29204d6d42697d)
- [Check out the detailed `TextGeneration` Pipeline documentation](https://github.com/neuralmagic/deepsparse/blob/main/docs/llms/text-generation-pipeline.md)

## **Roadmap**

Following these initial results, we are rapidly expanding our support for LLMs across the Neural Magic stack, including:

- **Productizing Sparse Fine Tuning**: Enable external users to apply the sparse fine-tuning to business datasets
- **Expanding Model Support**: Apply sparse fine-tuning results to Llama2 and Mistral models
- **Pushing to Higher Sparsity**: Improving our pruning algorithms to reach higher sparsity
- **Building General Sparse Model**: Create sparse model that can perform well on general tasks like OpenLLM leaderboard

## **Feedback / Roadmap Requests**

We are excited to add initial support for LLMs in the Neural Magic stack and plan to bring many ongoing improvements over the coming months. For questions or requests regarding LLMs, please reach out in any of the following channels:
- [Neural Magic Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
- [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues)
- [Contact Form](http://neuralmagic.com/contact/)
