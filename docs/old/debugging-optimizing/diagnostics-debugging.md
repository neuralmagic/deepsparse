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

# Logging Guidance for Diagnostics and Debugging

Unlike traditional software, debugging utilities available to the machine learning community are scarce. Complicated with deployment pipeline design issues, model weights, model architecture, and unoptimized models, debugging performance issues can be very dynamic in your data science ecosystem. Reviewing a log file can be your first line of defense in pinpointing performance issues with optimizing your inference.

The DeepSparse Engine ships with diagnostic logging so you can capture real-time monitoring information at model runtime and self-diagnose issues. If you are seeking technical support, we recommend capturing log information first, as described below. You can decide what to share, whether certain parts of the log or the entire content.  

**Note:** Our logs may reveal your inference network’s macro-architecture, including a general list of operators (such as convolution and pooling) and connections between them. Weights, trained parameters, or dataset parameters will not be captured. Consult Neural Magic’s various legal policies at [https://neuralmagic.com/legal/](https://neuralmagic.com/legal/) which include our privacy statement and software agreements. Your use of the software serves as your consent to these practices.

## Performance Tuning

An initial decision point to make in troubleshooting performance issues before enabling logs is whether to prevent threads from migrating from their cores. The default behavior is to disable thread binding (or pinning), allowing your OS to manage the allocation of threads to cores. There is a performance hit associated with this if the DeepSparseEngine is the main process running on your machine. If you want to enable thread binding for the possible performance benefit, set:

```bash
    NM_BIND_THREADS_TO_CORES=1
```

**Note 1:** If the DeepSparse Engine is not the only major process running on your machine, binding threads may hurt performance of the other major process(es) by monopolizing system resources.

**Note 2:** If you use OpenMP or TBB (Thread Building Blocks) in your application, then enabling thread binding may result in severe performance degradation due to conflicts between Neural Magic thread pool and OpenMP/TBB thread pools.

## Enabling Logs and Controlling the Amount of Logs Produced by the DeepSparse Engine

Logs are controlled by setting the `NM_LOGGING_LEVEL` environment variable.

Specify within your shell one of the following verbosity levels (in increasing order of verbosity:

`fatal, error, warn,` and `diagnose` with `diagnose` as a common default for all logs that will output to stderr:

```bash
    NM_LOGGING_LEVEL=diagnose 
    export NM_LOGGING_LEVEL
```

Alternatively, you can output the logging level by

```bash
    NM_LOGGING_LEVEL=diagnose <some command>
```

To enable diagnostic logs on a per-run basis, specify it manually before each script execution. For example, if you normally run:

```bash
    python run_model.py
```

Then, to enable diagnostic logs, run:

```bash
    NM_LOGGING_LEVEL=diagnose python run_model.py
```

To enable logging for your entire shell instance, execute within your shell:

```bash
    export NM_LOGGING_LEVEL=diagnose
```

By default, logs will print out to the stderr of your process. If you would like to output to a file, add `2> <name_of_log>.txt` to the end of your command.

## Parsing an Example Log

If you want to see an example log with `NM_LOGGING_LEVEL=diagnose`, a [truncated sample output](example-log.md) is provided at the end of this guide. It will show a super_resolution network, where Neural Magic only supports running 70% of it.

_Different portions of the log are explained below._

### Viewing the Whole Graph

Once a model is in our system, it is parsed to determine what operations it contains. Each operation is made a node and assigned a unique number Its operation type is displayed:

```bash
    Printing GraphViewer torch-jit-export:
    Node 0: Conv
    Node 1: Relu
    Node 2: Conv
    Node 3: Relu
    Node 4: Conv
    Node 5: Relu
    Node 6: Conv
    Node 7: Reshape
    Node 8: Transpose
    Node 9: Reshape
```

### Finding Supported Nodes for Our Optimized Engine

After the whole graph is loaded in, nodes are analyzed to determine whether they are supported by our optimized runtime engine. Notable "unsupported" operators are indicated by looking for `Unsupported [type of node]` in the log. For example, this is an unsupported Reshape node that produces a 6D tensor:

```bash
    [nm_ort 7f4fbbd3f740 >DIAGNOSE< unsupported /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/ops.cc:60] Unsupported Reshape , const shape greater than 5D
```

### Compiling Each Subgraph

Once all the nodes are located that are supported within the optimized engine, the graphs are split into maximal subgraphs and each one is compiled. ​To find the start of each subgraph compilation, look for `== Beginning new subgraph ==`. First, the nodes are displayed in the subgraph: ​

```bash
    Printing subgraph:
    Node 0: Conv 
    Node 1: Relu 
    Node 2: Conv 
    Node 3: Relu 
    Node 4: Conv 
    Node 5: Relu 
    Node 6: Conv
```

Simplifications are then performed on the graph to get it in an ideal state for complex optimizations, which are logged:

```bash
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:706] == Translating subgraph NM_Subgraph_1 to NM intake graph.
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:715] ( L1 graph 
    ( values:
      (10 float [ 1, 64, 224, 224 ])
      (11 float [ 1, 64, 224, 224 ])
      (12 float [ 1, 64, 224, 224 ])
      (13 float [ 1, 32, 224, 224 ])
      (14 float [ 1, 32, 224, 224 ])
      (15 float [ 1, 9, 224, 224 ])
      (9 float [ 1, 64, 224, 224 ])
      (conv1.bias float [ 64 ])
      (conv1.weight float [ 64, 1, 5, 5 ])
      (conv2.bias float [ 64 ])
      (conv2.weight float [ 64, 64, 3, 3 ])
      (conv3.bias float [ 32 ])
      (conv3.weight float [ 32, 64, 3, 3 ])
      (conv4.bias float [ 9 ])
      (conv4.weight float [ 9, 32, 3, 3 ])
      (input float [ 1, 1, 224, 224 ])
    )
    ( operations:
      (Constant conv1.bias (constant float [ 64 ]))
      (Constant conv1.weight (constant float [ 64, 1, 5, 5 ]))
      (Constant conv2.bias (constant float [ 64 ]))
      (Constant conv2.weight (constant float [ 64, 64, 3, 3 ]))
      (Constant conv3.bias (constant float [ 32 ]))
      (Constant conv3.weight (constant float [ 32, 64, 3, 3 ]))
      (Constant conv4.bias (constant float [ 9 ]))
      (Constant conv4.weight (constant float [ 9, 32, 3, 3 ]))
      (Input input (io 0))
      (Conv input -> 9 (conv kernel = [ 64, 1, 5, 5 ] bias = [ 64 ] padding = {{2, 2}, {2, 2}} strides = {1, 1}))
      (Elementwise 9 -> 10 (calc Relu))
      (Conv 10 -> 11 (conv kernel = [ 64, 64, 3, 3 ] bias = [ 64 ] padding = {{1, 1}, {1, 1}} strides = {1, 1}))
      (Elementwise 11 -> 12 (calc Relu))
      (Conv 12 -> 13 (conv kernel = [ 32, 64, 3, 3 ] bias = [ 32 ] padding = {{1, 1}, {1, 1}} strides = {1, 1}))
      (Elementwise 13 -> 14 (calc Relu))
      (Conv 14 -> 15 (conv kernel = [ 9, 32, 3, 3 ] bias = [ 9 ] padding = {{1, 1}, {1, 1}} strides = {1, 1}))
      (Output 15 (io 0))
    )
)
```

### Determining the Number of Cores and Batch Size

This log detail describes the batch size and number of cores that Neural Magic is optimizing against. Look for `== Compiling NM_Subgraph` as in this example:

```bash
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:723] == Compiling NM_Subgraph_1 with batch size 1 using 18 cores.
```

### Obtaining Subgraph Statistics

Locating  `== NM Execution Provider supports` shows how many subgraphs we compiled and what percentage of the network we managed to support running:

```bash
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:122] Created 1 compiled subgraphs.
[nm_ort 7f4fbbd3f740 >DIAGNOSE< validate_minimum_supported_fraction /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/utility/graph_util.cc:321] == NM Execution Provider supports 70% of the network
```

### Viewing Runtime Execution Times

​For each subgraph Neural Magic optimizes, the execution time is reported by `ORT NM EP compute_func:` for each run as follows:

```bash
​[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:265] ORT NM EP compute_func: 6.478 ms
```
