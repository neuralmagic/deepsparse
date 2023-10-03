# **DeepSparse LLMs**

Since our founding in 2018, Neural Magic has brought the power of sparsity and quantization to speed up deep learning models on CPUs.

We are pleased to announce initial support for generative models in DeepSparse, starting with Mosaic's MPT, including:
- Sparsity and quantization performance optimizations
- Internally managed KV-caching infrastructure
- Custom kernels for key generative inference operations for prefill and decode

In this overview, we will discuss:
- Current status of Neural Magic's LLM sparsity research
- Our roadmap over the next several months
- How to try text generation with DeepSparse

For usage details, see the [Text Generation User Guide](text-generation-pipeline.md)

## **Sparse LLM Research**

Sparsity is a powerful model compression technique, where weights are removed from the network with limited accuracy drop. For instance, Neural Magic, has successfully pruned CNN models like ResNet and Transformer models like BERT to 90%+, a >10x compression in model size. This compression creates opportunity for performance optimizations in DeepSparse, which is specifically built to accelerate NNs using sparsity.

Recently, we have been focused on adapting our sparsity techniques to decoder-only generative models. Currently, we can prune a fine-tuned version of [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) to ~60% sparsity with <1% accuracy drop (--- UPDATE: see our paper on arxiv ---) using a technique called Downstream Pruning, where we prune the network during the fine-tuning process.

> **Note**: We currently can induce sparsity during domain adaptation fine-tuning. Sparsifying a general model that performs well on general tasks like [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is an area of active research.

### **Downstream Pruning on Grade-School Math (GSM)**

Open-source LLMs like MPT and Llama2 are fine-tuned onto downstream datasets for two reasons:
* **Instruction Tuning**: fine-tune the model onto examples that show an LLM how to respond to user input in a helpful way
* **Domain Adaptation**: fine-tune the model onto examples that teach an LLM information it does not currently understand

An example of how domain adaptation is helpful is the [Grade-school math quetion answering dataset (GSM)](https://huggingface.co/datasets/gsm8k). 

GSM is a difficult task for LLMs, as evidenced by the poor performance of the MPT-7B model. By fine-tuning onto ~7k examples, however, we can boost the model's accuracy on the test set significantly:

--- UPDATE: insert chart showing accuracy ---

The key insight from our paper is that we can prune the network during the fine-tuning process! We apply the [SparseGPT pruning algorithm](https://arxiv.org/pdf/2301.00774.pdf) after fine-tuning and retrain for one extra epoch. The result is a 60% sparse-quantized model with <1% accuracy drop on GSM:

--- UPDATE: insert chart showing accuracy ---

With <1% accuracy drop, the 60% sparse-quantized model runs YYx faster in DeepSparse than the baseline model:

--- UPDATE: insert performance chart ---

### **How Is This Useful For Real World Use?**

The GSM dataset is a toy example demonstrating how to adapt LLMs to new tasks.

It it serves as an example of how LLMs can be adapted (with a relatively small <8k sample dataset) to solve tasks which the general pretrained model cannot. Given the treasure-troves of proprietary, domain-specific data held by companies, we expect to see many production models fine-tuned to enable more accurate, smaller models fit to business tasks.

Using Neural Magic, we can deploy these fine-tuned models performantly on CPUs!

> **Note**: The research code for Downstream Pruning will be pushed publicall SparseML over the comping weeks to enable you to apply this flow to your dataset.

## Roadmap

As mentioned above, we have only initial support for LLMs in DeepSparse. We are investing hevaily to expand our offering including:

* **Supporting Llama2**: Apply Downstream Pruning flow to Llama2 (and other models)
* **Productizing Downstream Pruning**: Enable external users to apply the Downstream Pruning to their datasets, enabling creation of fine-tuned LLMs for business use cases.
* **Pushing to Higher Sparsity**: Expanding our pruning techniques to push sparsity as high as possible without dropping accuracy to enable further performance speedups.
* **Building General Sparse Model**: Create sparse model that can perform well on general tasks like OpenLLM leaderboard and can be deployed directly.

## Try It Now

The following examples demonstrate how to use the trained MPT models on DeepSparse. Checkout the [user guide on `TextGeneration`](text-generation-pipeline.md) for more details on usage.

Make sure you have the nightly build of DeepSparse installed to run the examples.

```bash
pip install deepsparse-nightly[transformers]
```

### MPT-7B on GSM 

We can run inference on the 60% sparse-quantized MPT-7B GSM model ([available in SparseZoo](https://sparsezoo.neuralmagic.com/models/mpt-7b-gsm8k_mpt_pretrain-pruned50_quantized?showHidden=1)) using DeepSparse Pipelines:

```python
from deepsparse import TextGeneration

MODEL_PATH = "zoo:nlg/text_generation/mpt-7b/pytorch/huggingface/gsm8k/pruned50_quant-none"
pipeline = TextGeneration(model_path=MODEL_PATH)

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

### >> Natalia sold 48/2 = <<48/2=24>>24 clips in May.
### >> Natalia sold 48 + 24 = <<48+24=72>>72 clips altogether in April and May.
### >> #### 72
```

### **MPT-7B on Dolly-HHRLHF**

We have made a 50% sparse-quantized MPT-7B fine-tuned on the [Dolly-HHRLHF](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) instruction tuning dataset [available on SparseZoo](zoo:nlg/text_generation/mpt-7b/pytorch/huggingface/dolly/pruned50_quant-none).

***Caution: these models drop signficiant accuracy on general LLM evaluation tasks OpenLLM leaderboard*** and are meant to serve as demonstrations.

```python
from deepsparse import TextGeneration

MODEL_PATH = "zoo:nlg/text_generation/mpt-7b/pytorch/huggingface/gsm8k/pruned50_quant-none"
pipeline = TextGeneration(model_path=MODEL_PATH)

prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: what is Kubernetes? ### Response:"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

### >> Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management of containerized applications.
```

## **Feedback / Roadmap Requests**

We are excited to add initial support for LLMs in the Neural Magic stack and to bring ongoing improvements over the coming months.

For questions or requests regarding LLMs, please reach out in any of the following channels:
- [Neural Magic Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
- [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues)
- [Contact Form](http://neuralmagic.com/contact/)
