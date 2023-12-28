<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DeepSparse LLM Benchmarking with LLMPerf

![inter_token_vs_throughput](https://github.com/neuralmagic/deepsparse/assets/3195154/52f3a581-adaa-4853-a442-4f1a1a8d7324)

(LLMPerf)[https://github.com/ray-project/llmperf] is a tool for evaulation the performance of LLM APIs. In this example, we will use this project to benchmark the DeepSparse LLM Server using the OpenAI interface.

## Load test

The load test spawns a number of concurrent requests to the LLM API and measures the inter-token latency and generation throughput per request and across concurrent requests. The prompt that is sent with each request is of the format:

```
Randomly stream lines from the following text. Don't generate eos tokens:
LINE 1,
LINE 2,
LINE 3,
...
```

Where the lines are randomly sampled from a collection of lines from Shakespeare sonnets. Tokens are counted using the `LlamaTokenizer` regardless of which LLM API is being tested. This is to ensure that the prompts are consistent across different LLM APIs.

## Installation with DeepSparse

```
pip install deepsparse-nightly[llm]==1.7.0.20231210
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

## Simple Usage

Start up a server hosting TinyStories-1M with DeepSparse:
```
deepsparse.server --integration openai --task text-generation --model_path hf:mgoin/TinyStories-1M-ds
```

Then start the client benchmarking from within the `llmperf/` repo:
```
export OPENAI_API_KEY=dummy
export OPENAI_API_BASE="http://localhost:5543/v1"
python token_benchmark_ray.py \
--model "hf:mgoin/TinyStories-1M-ds" \
--mean-input-tokens 100 \
--stddev-input-tokens 50 \
--mean-output-tokens 100 \
--stddev-output-tokens 50 \
--max-num-completed-requests 50 \
--num-concurrent-requests 1 \
--llm-api openai
```

This should result in an output summary like this:

```
2023-12-28 15:10:30,691 INFO worker.py:1673 -- Started a local Ray instance.
52it [00:16,  3.24it/s]                                                                                                                                                         
\Results for token benchmark for hf:mgoin/TinyStories-1M-ds queried with the openai api.

inter_token_latency_s
    p25 = 0.005656395034778967
    p50 = 0.00691082744044463
    p75 = 0.008178272895754058
    p90 = 0.009280946066247402
    p95 = 0.020429442197616558
    p99 = 0.030548013546631083
    mean = 0.008013572023727383
    min = 0.0026352175821860633
    max = 0.03093951577320695
    stddev = 0.005740562313517516
ttft_s
    p25 = 0.05918681574985385
    p50 = 0.07706795702688396
    p75 = 0.11369983578333631
    p90 = 0.18972366366069765
    p95 = 0.22912864976096892
    p99 = 0.35286331018200157
    mean = 0.10145763676309098
    min = 0.030506487004458904
    max = 0.3802950750105083
    stddev = 0.07187234691400345
end_to_end_latency_s
    p25 = 0.5439230892225169
    p50 = 0.6968861420173198
    p75 = 0.923964643268846
    p90 = 1.1392052087234332
    p95 = 1.1703261025599203
    p99 = 1.287659268334974
    mean = 0.7156588349921199
    min = 0.1777444859035313
    max = 1.3783446750603616
    stddev = 0.2926796658407107
request_output_throughput_token_per_s
    p25 = 115.95619623824074
    p50 = 137.35373868635324
    p75 = 164.51010921901866
    p90 = 199.0789944243865
    p95 = 291.8986268349748
    p99 = 329.4090663122032
    mean = 147.8530523229102
    min = 31.846375045053573
    max = 363.71254109747105
    stddev = 63.041822008104916
number_input_tokens
    p25 = 84.0
    p50 = 103.0
    p75 = 133.0
    p90 = 174.79999999999998
    p95 = 191.94999999999996
    p99 = 217.13000000000008
    mean = 109.73076923076923
    min = 33
    max = 236
    stddev = 44.2138809649743
number_output_tokens
    p25 = 82.25
    p50 = 108.5
    p75 = 144.0
    p90 = 163.0
    p95 = 168.45
    p99 = 173.41000000000003
    mean = 105.96153846153847
    min = 15
    max = 178
    stddev = 46.68541424759001
Number Of Errored Requests: 0
Overall Output Throughput: 343.3687633083576
Number Of Completed Requests: 52
Completed Requests Per Minute: 194.43022532161083
```

## Advanced Usage

For a more realistic server benchmark, we will use a more useful LLM called (MiniChat-1.5-3B)[https://huggingface.co/neuralmagic/MiniChat-1.5-3B-pruned50-quant-ds] and enable continuous batching to do parallel decode steps.

First, we must make a config file to use continuous batching. Let's call it `config.yaml`:
```yaml
integration: openai
endpoints:
  - task: text_generation
    model: hf:neuralmagic/MiniChat-1.5-3B-pruned50-quant-ds
    kwargs:
      {"continuous_batch_sizes": [4]}
```

Then we can start up the server:
```
deepsparse.server --integration openai --config_file config.yaml
```

Finally let's make a more complex benchmark with 4 concurrent requests:
```
export OPENAI_API_KEY=dummy
export OPENAI_API_BASE="http://localhost:5543/v1"
python token_benchmark_ray.py \
--model "hf:neuralmagic/MiniChat-1.5-3B-pruned50-quant-ds" \
--mean-input-tokens 100 \
--stddev-input-tokens 50 \
--mean-output-tokens 200 \
--stddev-output-tokens 50 \
--max-num-completed-requests 50 \
--num-concurrent-requests 4 \
--timeout 600 \
--results-dir "result_outputs" \
--llm-api openai
```

Truncated output:
```
Number Of Errored Requests: 0
Overall Output Throughput: 50.65074661504567
Number Of Completed Requests: 50
Completed Requests Per Minute: 14.992820902332216
```

Since we saved the output to `result_outputs/`, we can also analyze it! See the code below and resulting image made from the above run:

```python
import pandas as pd

# path to the individual responses json file
df = pd.read_json('result_outputs/hf-neuralmagic-MiniChat-1-5-3B-pruned50-quant-ds_100_200_individual_responses.json')
valid_df = df[(df["error_code"] != "")]
print(valid_df.columns)

# End to end latency
all_token_latencies = valid_df['end_to_end_latency_s'].apply(pd.Series).stack()
p = all_token_latencies.plot.hist(title="End-to-end Latencies (s)", bins=30)
p.figure.savefig("e2e_latencies.png")
p.figure.clf()

# Inter-token latency
inter_token_latencies = valid_df['inter_token_latency_s'].apply(pd.Series).stack() * 1000
p = inter_token_latencies.plot.hist(title="Inter-token Latencies (ms)", bins=30)
p.figure.savefig("inter_token_latencies.png")
p.figure.clf()

# Time to first token
p = valid_df.plot.scatter(x="number_input_tokens", y="ttft_s", title="Number of Input Tokens vs. TTFT")
p.figure.savefig("time_to_first_token.png")
p.figure.clf()

# Time to first token
p = valid_df.plot.scatter(x="inter_token_latency_s", y="request_output_throughput_token_per_s", title="Inter-token Latency vs. Output Throughput")
p.figure.savefig("inter_token_vs_throughput.png")
p.figure.clf()
```
![e2e_latencies](https://github.com/neuralmagic/deepsparse/assets/3195154/34767a6b-dd4b-4101-bd77-b9f058dcaa43)
![inter_token_latencies](https://github.com/neuralmagic/deepsparse/assets/3195154/4e0ee366-9d9c-4a42-b2a0-e5f7676daad7)
![inter_token_vs_throughput](https://github.com/neuralmagic/deepsparse/assets/3195154/52f3a581-adaa-4853-a442-4f1a1a8d7324)
![time_to_first_token](https://github.com/neuralmagic/deepsparse/assets/3195154/f8956ad5-0c0f-440b-8a83-18ff4ed07453)


