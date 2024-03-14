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

import os
import time

from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from deepsparse import Context, Pipeline


os.environ["NM_BIND_THREADS_TO_CORES"] = "1"
INPUT_COL = "text"
dataset = load_dataset("ag_news", split="train[:3000]")
batch_size = 64
buckets = [64, 128, 256]
model_path = "./sparse-model/deployment/"

# TOKENIZE DATASET - (used to comptue buckets)
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
max_token_len = dataset[-1]["num_tokens"]

# SPLIT DATA INTO BATCHES
num_pad_items = batch_size - (dataset.num_rows % batch_size)
inputs = ([""] * num_pad_items) + dataset[INPUT_COL]
batches = []

for b_index_start in range(0, len(inputs), batch_size):
    batches.append(inputs[b_index_start : b_index_start + batch_size])

# RUN THROUPUT TESTING
print("\nCompiling models:")

tc_pipeline = Pipeline.create(
    task="zero_shot_text_classification",
    model_path=model_path,
    model_scheme="mnli",
    sequence_length=buckets,
    batch_size=batch_size,
    context=Context(num_streams=1),
)
print("\nRunning test:")
# run inferences on the datset
start = time.perf_counter()

predictions = []
for batch in tqdm(batches):
    predictions.append(
        tc_pipeline(sequences=batch, labels=["Sports", "Business", "Sci/Tech"])
    )

# flatten and remove padded predictions
predictions = [pred for sublist in predictions for pred in sublist.labels]
predictions = predictions[num_pad_items:]
end = time.perf_counter()

# compute throughput
total_time_executing = end - start
print(f"Total time: {total_time_executing}")
items_per_sec = len(predictions) / total_time_executing

print(f"Items Per Second: {items_per_sec}")
