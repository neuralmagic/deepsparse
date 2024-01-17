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

import time

from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

import torch
from datasets import load_dataset


model_path = "./dense-model/training/"
batch_size = 64

### SETUP DATASETS - in this case, we download ag_news
print("Setting up the dataset:")

INPUT_COL = "text"
dataset = load_dataset("ag_news", split="train[:3000]")

### TOKENIZE DATASETS - to sort dataset
tokenizer = AutoTokenizer.from_pretrained(model_path)


def pre_process_fn(examples):
    return tokenizer(
        examples[INPUT_COL],
        add_special_tokens=True,
        return_tensors="np",
        padding=False,
        truncation=False,
    )


dataset = dataset.map(pre_process_fn, batched=True)
dataset = dataset.add_column("num_tokens", list(map(len, dataset["input_ids"])))
dataset = dataset.sort("num_tokens")

### SPLIT DATA INTO BATCHES
hf_dataset = KeyDataset(dataset, INPUT_COL)

### RUN THROUGPUT TESTING
# load model
hf_pipeline = pipeline(
    "zero-shot-classification",
    model_path,
    batch_size=batch_size,
    device=("cuda:0" if torch.cuda.is_available() else "cpu"),
)

# run inferences
start = time.perf_counter()

predictions = []
for prediction in hf_pipeline(
    hf_dataset, candidate_labels=["Sports", "Business", "Sci/Tech"]
):
    predictions.append(prediction)

# torch.cuda.synchronize()

end = time.perf_counter()

# compute throughput
total_time_executing = end - start
items_per_sec = len(predictions) / total_time_executing

print(f"Total time: {total_time_executing}")
print(f"Items Per Second: {items_per_sec}")
