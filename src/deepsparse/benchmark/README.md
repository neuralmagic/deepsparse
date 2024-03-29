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

## 📜 Benchmarking ONNX Models

`deepsparse.benchmark` is a command-line (CLI) tool for benchmarking the DeepSparse Engine with ONNX models. The tool will parse the arguments, download/compile the network into the engine, generate input tensors, and execute the model depending on the chosen scenario. By default, it will choose a multi-stream or asynchronous mode to optimize for throughput.

### Quickstart

After `pip install deepsparse`, the benchmark tool is available on your CLI. For example, to benchmark a dense BERT ONNX model fine-tuned on the MNLI dataset where the model path is the minimum input required to get started, run:

```
deepsparse.benchmark zoo:nlp/text_classification/bert-base/pytorch/huggingface/mnli/base-none
```
__ __
### Usage

In most cases, good performance will be found in the default options so it can be as simple as running the command with a SparseZoo model stub or your local ONNX model. However, if you prefer to customize benchmarking for your personal use case, you can run `deepsparse.benchmark -h` or with `--help` to view your usage options:

CLI Arguments:
```
positional arguments:

        model_path                    Path to an ONNX model file or SparseZoo model stub.

optional arguments:

        -h, --help                    show this help message and exit.

        -b BATCH_SIZE, --batch_size BATCH_SIZE
                                        The batch size to run the analysis for. Must be
                                        greater than 0.

        -shapes INPUT_SHAPES, --input_shapes INPUT_SHAPES
                                        Override the shapes of the inputs, i.e. -shapes
                                        "[1,2,3],[4,5,6],[7,8,9]" results in input0=[1,2,3]
                                        input1=[4,5,6] input2=[7,8,9].

        -ncores NUM_CORES, --num_cores NUM_CORES
                                        The number of physical cores to run the analysis on,
                                        defaults to all physical cores available on the system.

        -s {async,sync,elastic}, --scenario {async,sync,elastic}
                                        Choose between using the async, sync and elastic
                                        scenarios. Sync and async are similar to the single-
                                        stream/multi-stream scenarios. Elastic is a newer
                                        scenario that behaves similarly to the async scenario
                                        but uses a different scheduling backend. Default value
                                        is async.

        -t TIME, --time TIME            
                                        The number of seconds the benchmark will run. Default
                                        is 10 seconds.

        -w WARMUP_TIME, --warmup_time WARMUP_TIME
                                        The number of seconds the benchmark will warmup before
                                        running.Default is 2 seconds.

        -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                                        The number of streams that will submit inferences in
                                        parallel using async scenario. Default is
                                        automatically determined for given hardware and may be
                                        sub-optimal.

        -pin {none,core,numa}, --thread_pinning {none,core,numa}
                                        Enable binding threads to cores ('core' the default),
                                        threads to cores on sockets ('numa'), or disable
                                        ('none').

        -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                                        Inference engine backend to run eval on. Choices are
                                        'deepsparse', 'onnxruntime'. Default is 'deepsparse'.

        -q, --quiet                     Lower logging verbosity.

        -x EXPORT_PATH, --export_path EXPORT_PATH
                                        Store results into a JSON file.
```
💡**PRO TIP**💡: save your benchmark results in a convenient JSON file!

Example CLI command for benchmarking an ONNX model from the SparseZoo and saving the results to a `benchmark.json` file:

```
deepsparse.benchmark zoo:nlp/text_classification/bert-base/pytorch/huggingface/mnli/base-none -x benchmark.json
```
Output of the JSON file:

![alt text](./img/json_output.png)

#### Sample CLI Argument Configurations

To run a sparse FP32 MobileNetV1 at batch size 16 for 10 seconds for throughput using 8 streams of requests:

```
deepsparse.benchmark zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate --batch_size 16 --time 10 --scenario async --num_streams 8
```

To run a sparse quantized INT8 BERT at batch size 1 for latency:

```
deepsparse.benchmark zoo:nlp/question_answering/bert-large/pytorch/huggingface/squad/pruned90_quant-none --batch_size 1 --scenario sync
```
__ __
### ⚡ Inference Scenarios

#### Synchronous (Single-stream) Scenario

Set by the `--scenario sync` argument, the goal metric is latency per batch (ms/batch). This scenario submits a single inference request at a time to the engine, recording the time taken for a request to return an output. This mimics an edge deployment scenario.

The latency value reported is the mean of all latencies recorded during the execution period for the given batch size.

#### Asynchronous (Multi-stream) Scenario

Set by the `--scenario async` argument, the goal metric is throughput in items per second (i/s). This scenario submits `--num_streams` concurrent inference requests to the engine, recording the time taken for each request to return an output. This mimics a model server or bulk batch deployment scenario.

The throughput value reported comes from measuring the number of finished inferences within the execution time and the batch size.

#### Example Benchmarking Output of Synchronous vs. Asynchronous

**BERT 3-layer FP32 Sparse Throughput**

No need to add *scenario* argument since `async` is the default option:
```
deepsparse.benchmark zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_83
[INFO benchmark_model.py:202 ] Thread pinning to cores enabled
DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. version: 0.10.0 (9bba6971) (optimized) (system=avx512, binary=avx512)
[INFO benchmark_model.py:247 ] deepsparse.engine.Engine:
        onnx_file_path: /home/mgoin/.cache/sparsezoo/c89f3128-4b87-41ae-91a3-eae8aa8c5a7c/model.onnx
        batch_size: 1
        num_cores: 18
        scheduler: Scheduler.multi_stream
        cpu_avx_type: avx512
        cpu_vnni: False
[INFO            onnx.py:176 ] Generating input 'input_ids', type = int64, shape = [1, 384]
[INFO            onnx.py:176 ] Generating input 'attention_mask', type = int64, shape = [1, 384]
[INFO            onnx.py:176 ] Generating input 'token_type_ids', type = int64, shape = [1, 384]
[INFO benchmark_model.py:264 ] num_streams default value chosen of 9. This requires tuning and may be sub-optimal
[INFO benchmark_model.py:270 ] Starting 'async' performance measurements for 10 seconds
Original Model Path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_83
Batch Size: 1
Scenario: multistream
Throughput (items/sec): 83.5037
Latency Mean (ms/batch): 107.3422
Latency Median (ms/batch): 107.0099
Latency Std (ms/batch): 12.4016
Iterations: 840
```

**BERT 3-layer FP32 Sparse Latency**

To select a *synchronous inference scenario*, add `-s sync`:

```
deepsparse.benchmark zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_83 -s sync
[INFO benchmark_model.py:202 ] Thread pinning to cores enabled
DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. version: 0.10.0 (9bba6971) (optimized) (system=avx512, binary=avx512)
[INFO benchmark_model.py:247 ] deepsparse.engine.Engine:
        onnx_file_path: /home/mgoin/.cache/sparsezoo/c89f3128-4b87-41ae-91a3-eae8aa8c5a7c/model.onnx
        batch_size: 1
        num_cores: 18
        scheduler: Scheduler.single_stream
        cpu_avx_type: avx512
        cpu_vnni: False
[INFO            onnx.py:176 ] Generating input 'input_ids', type = int64, shape = [1, 384]
[INFO            onnx.py:176 ] Generating input 'attention_mask', type = int64, shape = [1, 384]
[INFO            onnx.py:176 ] Generating input 'token_type_ids', type = int64, shape = [1, 384]
[INFO benchmark_model.py:270 ] Starting 'sync' performance measurements for 10 seconds
Original Model Path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_3layers-aggressive_83
Batch Size: 1
Scenario: singlestream
Throughput (items/sec): 62.1568
Latency Mean (ms/batch): 16.0732
Latency Median (ms/batch): 15.7850
Latency Std (ms/batch): 1.0427
Iterations: 622
```

## 📜 Benchmarking Pipelines
Expanding on the model benchmarking script, `deepsparse.benchmark_pipeline` is a tool for benchmarking end-to-end inference, including pre and post processing. The script can generate fake input data based on the pipeline's input schema, or load it from a local folder. The pipeline then runs pre-processing, engine inference and post-processing. Benchmarking results are reported by section, useful for identifying bottlenecks. 

### Usage 
Input arguments are the same as the Engine benchmarker, but with two additions:

```
positional arguments:
  task_name             Type of pipeline to run(i.e "text_generation")

optional arguments:
  -c INPUT_CONFIG, --input_config INPUT_CONFIG
                        JSON file containing schema for input data
```

The `input_config` argument is a path to a json file specifying details on the input schema to the pipeline, detailed below.

### Configuring Pipeline Inputs

Inputs to the pipeline are configured through a json config file. The `data_type` field should be set to `"dummy"` if passing randomly generated data through the pipeline, and `"real"` if passing in data from files.

#### Dummy Input Configuration
An example dummy input configuration is shown below.
* `gen_sequence_length`: number of characters to generate for pipelines that take text input
* `input_image_shape`: configures image size for pipelines that take image input, must be 3 dimmensional with channel as the last dimmension

```json
{
    "data_type": "dummy",
    "gen_sequence_length": 100,
    "input_image_shape": [500,500,3],
    "pipeline_kwargs": {},
    "input_schema_kwargs": {}
} 
```

#### Real Input Configuration
An example real input configuration is shown below.
* `data_folder`: path to local folder of input data, should contain text or image files
* `recursive_search`: whether to recursively search through `data_folder` for files
* `max_string_length`: maximum characters to read from each file containing text data, -1 for no max length

```json
{
    "data_type": "real",
    "data_folder": "/home/sadkins/imagenette2-320/",
    "recursive_search": true,
    "max_string_length": -1,
    "pipeline_kwargs": {},
    "input_schema_kwargs": {}
} 
```

#### Keyword Arguments
Additional arguments to the pipeline or input_schema can be added to the `pipeline_kwargs` and `input_schema_kwargs` fields respectively. For instance, to pass class_names to a YOLO pipeline and conf_thres to the input schema
```json
{
    "data_type": "dummy",
    "input_image_shape": [500,500,3],
    "pipeline_kwargs": {
        "class_names": ["classA", "classB"]
    },
    "input_schema_kwargs": {
        "conf_thres": 0.7
    }
} 
```

### Example Usage

Running ResNet image classification for 30 seconds with a batch size of 32:
```
deepsparse.benchmark_pipeline image_classification zoo:cv/classification/resnet_v1-50_2x/pytorch/sparseml/imagenet/base-none -c config.json -t 60 -b 32
```

Running CodeGen text generation for 30 seconds asynchronously 
```
deepsparse.benchmark_pipeline text_generation zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/pruned50-none -c config.json -t 30 -s async
```
### Example Output
Command:
```
deepsparse.benchmark_pipeline text_classification zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned90-none -c config.json
```
config.json:
```json
{
    "data_type": "real",
    "gen_sequence_length": 1000,
    "data_folder": "/home/sadkins/text_data/",
    "recursive_search": true,
    "max_string_length": -1
}
```

Output:
```
Batch Size: 1
Scenario: sync
Iterations: 955
Total Runtime: 10.0090
Throughput (items/sec): 95.4137
Processing Time Breakdown: 
     total_inference: 99.49%
     pre_process: 25.70%
     engine_forward: 72.56%
     post_process: 1.03%
Mean Latency Breakdown (ms/batch): 
     total_inference: 10.4274
     pre_process: 2.6938
     engine_forward: 7.6051
     post_process: 0.1077
```

Command:
```
deepsparse.benchmark_pipeline text_generation zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none -c config.json -t 60
```
config.json:
```json
{
    "data_type": "dummy",
    "gen_sequence_length": 100,
    "pipeline_kwargs": {},
    "input_schema_kwargs": {}
} 
```

Output:
```
Batch Size: 1
Scenario: sync
Iterations: 6
Total Runtime: 62.8005
Throughput (items/sec): 0.0955
Processing Time Breakdown: 
     total_inference: 100.00%
     pre_process: 0.00%
     engine_forward: 99.98%
     post_process: 0.01%
     engine_prompt_prefill: 5.83%
     engine_prompt_prefill_single: 0.09%
     engine_token_generation: 93.64%
     engine_token_generation_single: 0.09%
Mean Latency Breakdown (ms/batch): 
     total_inference: 20932.4786
     pre_process: 0.9729
     engine_forward: 20930.2190
     post_process: 1.2150
     engine_prompt_prefill: 1220.7037
     engine_prompt_prefill_single: 19.0412
     engine_token_generation: 19603.0353
     engine_token_generation_single: 19.1170
```
