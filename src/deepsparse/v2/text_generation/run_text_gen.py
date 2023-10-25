# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from deepsparse.transformers.pipelines.text_generation import TextGenerationInput
from deepsparse.v2.text_generation.pipeline import TextGenerationPipeline
from huggingface_hub import snapshot_download


def create_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_ground_truth(prompt):
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    tokenizer = create_tokenizer("roneneldan/TinyStories-1M")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    out = model(input_ids=input_ids)
    prompt_logits = out.logits.detach().numpy()
    return prompt_logits


engine_kwargs = {"engine_type": "onnxruntime"}
prompt = "Hello!"
model_path = "hf:mgoin/TinyStories-1M-deepsparse"
pipeline = TextGenerationPipeline(model_path)
input_values = TextGenerationInput(prompt=prompt)
logits = pipeline(input_values)
ground_truth = get_ground_truth(prompt)

for i in range(ground_truth.shape[1]):
    print(ground_truth[0, i, :])
    print(logits[0, i, :])
    print("\n")
print("All Close?", np.allclose(logits, ground_truth, atol=0.0001))
