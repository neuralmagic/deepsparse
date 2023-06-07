from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import time
import torch
model_path = "./dense-model/training/"
batch_size = 64

### SETUP DATASETS - in this case, we download ag_news
print("Setting up the dataset:")

INPUT_COL = "text"
dataset = load_dataset("ag_news", split="train[:3000]")

### TOKENIZE DATASETS - to sort dataset
tokenizer = AutoTokenizer.from_pretrained(model_path)

def pre_process_fn(examples):
    return tokenizer(examples[INPUT_COL], add_special_tokens=True, return_tensors="np", padding=False, truncation=False)

dataset = dataset.map(pre_process_fn, batched=True)
dataset = dataset.add_column("num_tokens", list(map(len, dataset["input_ids"])))
dataset = dataset.sort("num_tokens")

### SPLIT DATA INTO BATCHES
hf_dataset = KeyDataset(dataset, INPUT_COL)

### RUN THROUGPUT TESTING
# load model
hf_pipeline = pipeline("zero-shot-classification", model_path, batch_size=batch_size,device=("cuda:0" if torch.cuda.is_available() else "cpu"), )

# run inferences
start = time.perf_counter()

predictions = []
for prediction in hf_pipeline(hf_dataset,candidate_labels=['Sports', 'Business', 'Sci/Tech']):
    predictions.append(prediction)

# torch.cuda.synchronize()

end = time.perf_counter()

# compute throughput
total_time_executing = end - start
items_per_sec = len(predictions) / total_time_executing

print(f"Total time: {total_time_executing}")
print(f"Items Per Second: {items_per_sec}")
