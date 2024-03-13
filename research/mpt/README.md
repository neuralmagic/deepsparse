*LAST UPDATED: 11/24/2023*

# **Sparse Fine-Tuned LLMs With DeepSparse**

DeepSparse has support for performant inference of sparse large language models, starting with Mosaic's MPT and Meta's Llama 2.
Check out our paper [Sparse Fine-tuning for Inference Acceleration of Large Language Models](https://arxiv.org/abs/2310.06927)

In this research overview, we will discuss:
1. [Our Sparse Fine-Tuning Research](#sparse-finetuning-research)
2. [How to Try Text Generation With DeepSparse](#try-it-now)

## **Sparse Fine-Tuning Research**

We show that MPT-7B and Llama-2-7B can be pruned to ~60% sparsity with INT8 quantization (and 70% sparsity without quantization), with no accuracy drop, using a technique called **Sparse Fine-Tuning**, where we prune the network during the fine-tuning process.

When running the pruned network with DeepSparse, we can accelerate inference by ~7x over the dense-FP32 baseline!

### **Sparse Fine-Tuning on Grade-School Math (GSM)**

Training LLMs consists of two steps. First, the model is pre-trained on a very large corpus of text (typically >1T tokens). Then, the model is adapted for downstream use by continuing training with a much smaller high-quality curated dataset. This second step is called fine-tuning.

Fine-tuning is useful for two main reasons:
1. It can teach the model *how to respond* to input (often called **instruction tuning**).
2. It can teach the model *new information* (often called **domain adaptation**).

An example of how domain adaptation is helpful in solving the [Grade-school math (GSM) dataset](https://huggingface.co/datasets/gsm8k). GSM is a set of grade school word problems and a notoriously difficult task for LLMs, as evidenced by the 0% zero-shot accuracy of MPT-7B. By fine-tuning with a very small set of ~7k training examples, however, we can boost the model's accuracy on the test set to 28.2%.

The key insight from [our paper](https://arxiv.org/abs/2310.06927) is that we can prune the network during the fine-tuning process. We apply [SparseGPT](https://arxiv.org/pdf/2301.00774.pdf) to prune the network after dense fine-tuning and retrain for 2 epochs with L2 distillation. The result is a 60% sparse-quantized model with no accuracy drop on GSM8k runs 7x faster than the dense baseline with DeepSparse!

<div align="center">
    <img src="https://github.com/neuralmagic/deepsparse/assets/3195154/f9a86726-12f5-4926-8d8c-668c449faa84" width="60%" ALT="Sparse Fine-Tuned LLMs on GSM8k"/>
</div>

- [See the paper on Arxiv](https://arxiv.org/abs/2310.06927).
- [See our Llama 2 expansion blog on the initial paper](https://neuralmagic.com/blog/fast-llama-2-on-cpus-with-sparse-fine-tuning-and-deepsparse/).

### **How Is This Useful For Real-World Use?**

While GSM is a "toy" math dataset, it serves as an example of how LLMs can be adapted to solve tasks that the general pre-trained model cannot. Given the treasure troves of domain-specific data held by companies, we expect to see many production models fine-tuned to create more accurate models fit to business tasks. Using Neural Magic, you can deploy these fine-tuned models performantly on CPUs!

## Try It Now

Install the DeepSparse Nightly build (requires Linux):

```bash
pip install -U deepsparse-nightly[llm]
```

The models generated in the paper are hosted on [SparseZoo](https://sparsezoo.neuralmagic.com/?ungrouped=true&sort=null&datasets=gsm8k) and [Hugging Face](https://huggingface.co/collections/neuralmagic/sparse-finetuning-mpt-65241d875b29204d6d42697d). 

We can run inference on the models using DeepSparse's `TextGeneration` Pipeline:

```python
from deepsparse import TextGeneration

pipeline = TextGeneration(model_path="zoo:llama2-7b-gsm8k_llama2_pretrain-pruned60_quantized")

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

#### Other Resources
- [Check out all the GSM models on SparseZoo](https://sparsezoo.neuralmagic.com/?datasets=gsm8k&ungrouped=true).
- [Try out the live demo on Hugging Face Spaces](https://huggingface.co/spaces/neuralmagic/sparse-mpt-7b-gsm8k) and view the [collection of paper, demos, and models](https://huggingface.co/collections/neuralmagic/sparse-finetuning-mpt-65241d875b29204d6d42697d).
- [Check out the detailed `TextGeneration` Pipeline documentation](https://github.com/neuralmagic/deepsparse/blob/main/docs/llms/text-generation-pipeline.md).

## **Roadmap**

Following these initial results, we are rapidly expanding our support for LLMs across the Neural Magic stack, including:

- **Productizing Sparse Fine-Tuning**: Enable external users to apply the sparse fine-tuning to business datasets.
- **Expanding Model Support**: Apply sparse fine-tuning results to Mistral models.
- **Pushing to Higher Sparsity**: Improving our pruning algorithms to reach higher sparsity.
- **Building General Sparse Model**: Create a sparse model that can perform well on general tasks like OpenLLM leaderboard.

## **Feedback / Roadmap Requests**

We are excited to add initial support for LLMs in the Neural Magic stack and plan to bring many ongoing improvements over the coming months. For questions or requests regarding LLMs, reach out through any of the following channels:
- [Neural Magic Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
- [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues)
- [Contact Form](http://neuralmagic.com/contact/)
