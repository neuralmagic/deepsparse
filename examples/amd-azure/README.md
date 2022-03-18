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

# Multi-Process Throughput Benchmarking for AMD Systems

The command-line (CLI) tool `multi-process-benchmark.py` allows users to measure the performance using multiple separate processes in parallel. This is ideal for measuring performance on systems that have multiple sockets or multiple L3 caches, like AMD's Milan processor.

## Quickstart

Once users clone the DeepSparse Engine and installation requirements for `examples/amd-azure` the CLI script will be available.
```bash
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/amd-azure
```

To run parallel inferences of a sparse FP32 MobileNetV1 on each CCX of a Microsoft Azure HB120\_v3 at batch size 16 for 30 seconds for throughput,
the following command should be run from inside the `examples/amd-azure` directory:
```bash
python multi_process_benchmark.py zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate ./azure_hb120_v3.json --batch_size 16 --time 30
```

## What does it do?

The tool will parse the arguments, download/compile the network into the DeepSparse Engine, generate input tensors, and execute the model depending on the chosen parameters.

The number of executions is a result of setting the time duration to run executions for, set by the `-t` argument in seconds.

The throughput value reported comes from measuring the number of finished inferences within the execution time and the batch size.

## Topology JSON files

For this benchmarking script, users must specify the topology of their system with a JSON file. This file includes a list of lists of cores. 

One list of cores will contain the processor IDs that one of the worker processes will run on, and should reflect the topology of the system. For performance, one list of cores in the JSON topology file should contain the list of cores that are on the same socket, are on the same NUMA node, or share the same L3 cache. 

The `/examples/amd-azure` directory contains two example JSON files that can be used:

- `azure_hb120_v3.json`, which is suitable for use on a Microsoft Azure HB120\_V3 instance. You may notice that not every process will use the same number of cores when using this topology. This is because some of the CCXs on this instance type have some cores dedicated to running the hypervisor.
- `amd_epyc_7713.json`, which is suitable for a two-socket system with AMD EPYC 7713 processors. This file will also work for a one-socket system if the proper parameter for `nstreams` is passed into `multi_process_benchmark.py`.

## Usage

First, install the requirements using
```bash
pip install -r requirements.txt
```
`multi_process_benchmark.py` uses `py-libnuma` to control the CPU affinity and memory policy of each individual stream to optimize performance.

In most cases, good performance will be found in the default options so it can be as simple as running the command with a [Neural Magic SparseZoo](https://sparsezoo.neuralmagic.com) model stub or your local ONNX model.

```bash
python multi_process_benchmark.py <path/to/model> <path/to/topology/file>
```

Executing `multi_process_benchmark.py -h` or `--help` provides the following usage options:

```
usage: python multi_process_benchmark.py [-h] [-b BATCH_SIZE] [-nstreams NUM_STREAMS] [-shapes INPUT_SHAPES] [-t TIME] [-w WARMUP_TIME]
                                  [-pin {none,core,numa}] [-q]
                                  model_path topology_file

Arguments for benchmarking ONNX models in the DeepSparse Engine:

positional arguments:
  model_path            Path to an ONNX model file or SparseZoo model stub.
  topology_file         Path to a JSON file describing the topology of the system. This JSON file will contain a list of lists of cores. The ith
                        such list will contain the cores that will be used by the ith process. As such there must be at least nstreams lists of
                        cores.

optional arguments:
  -h, --help            Provide help message and exit.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for. Must be greater than 0.
  -nstreams NUM_STREAMS, --num_streams NUM_STREAMS
                        The number of processes that will run inferences in parallel.
  -shapes INPUT_SHAPES, --input_shapes INPUT_SHAPES
                        Override the shapes of the inputs, i.e., -shapes "[1,2,3],[4,5,6],[7,8,9]" results in input0=[1,2,3] input1=[4,5,6]
                        input2=[7,8,9].
  -t TIME, --time TIME  The number of seconds the benchmark will run; default is 20 seconds.
  -w WARMUP_TIME, --warmup_time WARMUP_TIME
                        The number of seconds the benchmark will warm up before running and cool down after running; default is 5 seconds.
  -pin {none,core,numa}, --thread_pinning {none,core,numa}
                        Enable binding threads to cores ('core' the default), threads to cores on sockets ('numa'), or disable ('none')
  -q, --quiet           Lower logging verbosity and suppress output from worker processes.
```

## Example of benchmarking output

**BERT 12-layer FP32 Sparse Throughput:**

```
python3 multi_process_benchmark.py "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98" ./amd_epyc_7713.json -b 16  -pin core --input_shapes='[1,128],[1,128],[1,128]' -q
DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. version: 0.12.0 (5ecb02cd) (optimized) (system=avx2, binary=avx2)
Original Model Path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98
Batch Size: 16
Throughput (items/sec): 965.6019
Latency Mean (ms/batch): 265.0954
Latency Median (ms/batch): 263.3038
Latency Std (ms/batch): 10.2972
Iterations: 1216
```
