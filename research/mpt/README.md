# **Sparse Finetuned LLMs with DeepSparse**

DeepSparse has support for performant inference of sparse large language models, starting with Mosaic's MPT. 

In this overview, we will discuss:
1. [Current status of our sparse fine-tuning research](#sparse-fine-tuning-research)
2. [How to try text generation with DeepSparse](#try-it-now)

For detailed usage instructions, [see the text generation user guide](text-generation-pipeline.md).

## **Sparse Finetuning Research**

Sparsity is a powerful model compression technique, where weights are removed from the network with limited accuracy drop. 

We show that MPT-7B can be pruned to ~60% sparsity with INT8 quantization, without loss, using a technique called **Sparse Finetuning**, where we prune the network during the fine-tuning process.

### **Sparse Finetuning on Grade-School Math (GSM)**

Open-source LLMs are typically fine-tuned onto downstream datasets for two reasons:
* **Instruction Tuning**: show the LLM examples of how to respond to human input or prompts properly
* **Domain Adaptation**: show the LLM examples with information it does not currently understand

An example of how domain adaptation is helpful is solving the [Grade-school math (GSM) dataset](https://huggingface.co/datasets/gsm8k). GSM is a set of grade school word problems and a notoriously difficult task for LLMs, as evidenced by the 0% zero-shot accuracy of MPT-7B-base. By fine-tuning with a very small set of ~7k training examples, however, we can boost the model's accuracy on the test set to 28.2%.

The key insight from our paper is that we can prune the network during the fine-tuning process! We apply [SparseGPT](https://arxiv.org/pdf/2301.00774.pdf) to prune the network after fine-tuning and retrain for one extra epoch. The result is a 60% sparse-quantized model with limited accuracy drop on GSM8k runs 6.7x faster than the dense baseline with DeepSparse!

### **How Is This Useful For Real World Use?**

While GSM is a "toy" math dataset, it serves as an example of how LLMs can be adapted to solve tasks which the general pretrained model cannot. Given the treasure-troves of domain-specific data held by companies, we expect to see many production models fine-tuned to create more accurate models fit to business tasks. Using Neural Magic, you can deploy these fine-tuned models performantly on CPUs!

## Try It Now

Install the DeepSparse Nightly build (requires Linux):

```bash
pip install deepsparse-nightly[transformers]
```

### MPT-7B on GSM 

We can run inference on the 60% sparse-quantized MPT-7B GSM model using DeepSparse's `TextGeneration` Pipeline:

```python
from deepsparse import TextGeneration

MODEL_PATH = "zoo:nlg/text_generation/mpt-7b/pytorch/huggingface/gsm8k/pruned60_quant-none"
pipeline = TextGeneration(model_path=MODEL_PATH)

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

### >> Natalia sold 48/2 = <<48/2=24>>24 clips in May.
### >> Natalia sold 48 + 24 = <<48+24=72>>72 clips altogether in April and May.
### >> #### 72
```

It is also possible to run models directly from Hugging Face by prepending `"hf:"` to a model id, such as:

```python
from deepsparse import TextGeneration

MODEL_PATH = "hf:neuralmagic/mpt-7b-gsm8k-pruned60-quant"
pipeline = TextGeneration(model_path=MODEL_PATH)

prompt = "Question: Marty has 100 centimeters of ribbon that he must cut into 4 equal parts. Each of the cut parts must be divided into 5 equal parts. How long will each final cut be?"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

### >> The 100-centimeter ribbon is cut into 4 equal parts in pieces of 100 / 4 = <<100/4=25>>25 cm
### >> From each piece of 25 cm, he gets 5 equal parts of 25 / 5 = <<25/5=5>>5 cm each.
### >> #### 5
```

#### Other Resources
- [Check out all the models on SparseZoo](https://sparsezoo.neuralmagic.com/models/mpt-7b-gsm8k_mpt_pretrain-pruned60_quantized)
- [Try out the live demo on Hugging Face Spaces](https://huggingface.co/spaces/neuralmagic/sparse-mpt-7b-gsm8k-deepsparse) and view the [collection of paper, demos, and models](https://huggingface.co/collections/neuralmagic/sparse-finetuning-mpt-65241d875b29204d6d42697d)

### **MPT-7B on Dolly-HHRLHF**

We have also made a 50% sparse-quantized MPT-7B fine-tuned on [Dolly-hhrlhf](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) available on SparseZoo. We can run inference with the following:

```python
from deepsparse import TextGeneration

MODEL_PATH = "zoo:nlg/text_generation/mpt-7b/pytorch/huggingface/dolly/pruned50_quant-none"
pipeline = TextGeneration(model_path=MODEL_PATH)

prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: what is Kubernetes? ### Response:"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

### >> Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management of containerized applications.
```

## **Feedback / Roadmap Requests**

We are excited to add initial support for LLMs in the Neural Magic stack and plan to bring many ongoing improvements over the coming months. For questions or requests regarding LLMs, please reach out in any of the following channels:
- [Neural Magic Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
- [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues)
- [Contact Form](http://neuralmagic.com/contact/)
