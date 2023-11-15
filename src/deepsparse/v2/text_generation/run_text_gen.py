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
import asyncio
import time

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from deepsparse.transformers.pipelines.text_generation import TextGenerationInput
from deepsparse.v2.text_generation.pipeline import TextGenerationPipeline
from deepsparse.v2.utils.state import InferenceState


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


from transformers import GenerationConfig


# output = pipeline(input_value)
# print(out)
"""
async def func():
    inference_state = InferenceState()
    inference_state.create_state({})
    pipeline_state = pipeline.pipeline_state

    output = await pipeline.run_async(input_value, pipeline_state=pipeline_state, inference_state=inference_state)
    return output
print(asyncio.run(func()))
"""

model_path = "hf:mgoin/TinyStories-1M-deepsparse"
pipeline = TextGenerationPipeline(model_path, prompt_sequence_length=3, engine_kwargs={"engine_type": "onnxruntime"})


def run_requests():
    prompts = [["Hello there!", "How are you?"]]
    outputs = []
    for i in range(len(prompts)):
        input_value = TextGenerationInput(
            prompt=prompts[i],
            generation_kwargs={
                "do_sample": False,
                "max_length": 20,
            },
        )
        output = pipeline(input_value)
        yield output


output = run_requests()
for x in output:
    for g in x.generations:
        print("\n")
        print(g)
