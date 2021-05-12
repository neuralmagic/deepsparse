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

# Example Log, Verbose Level = diagnose

The following is an example log with `NM_LOGGING_LEVEL=diagnose` running a super_resolution network, where we only support running 70% of it. Different portions of the log are explained in [Parsing an Example Log](diagnostics-debugging.md#parsing-an-example-log).

```bash
onnx_filename : test-models/cv-resolution/super_resolution/none-bsd300-onnx-repo/model.onnx
[     INFO     neuralmagic.py: 112 -   neuralmagic_create() ] Construct network from ONNX = test-models/cv-resolution/super_resolution/none-bsd300-onnx-repo/model.onnx
NeuralMagic WAND version: 1.0.0.96ce2f6cb23b8ab377012ed9ef38d3da3b9f5313 (optimized) (system=avx512, binary=avx512)
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:104] == NMExecutionProvider::GetCapability ==
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
​
[nm_ort 7f4fbbd3f740 >DIAGNOSE< unsupported /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/ops.cc:60] Unsupported Reshape , const shape greater than 5D
[nm_ort 7f4fbbd3f740 >DIAGNOSE< construct_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:595] == Constructing subgraphs from graph info
[nm_ort 7f4fbbd3f740 >WARN<  construct_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:604] Cannot support patterns, defaulting to non-pattern-matched subgraphs
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:644] == Beginning new subgraph ==
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:667] Runtime inputs for subgraph:
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:679]     input (required)
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:684] Printing subgraph:
Node 0: Conv 
Node 1: Relu 
Node 2: Conv 
Node 3: Relu 
Node 4: Conv 
Node 5: Relu 
Node 6: Conv 
​
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
​
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:723] == Compiling NM_Subgraph_1 with batch size 1 using 18 cores.
[7f4fbbd3f740 >DIAGNOSE< allocate_buffers_pass ./src/include/wand/engine/units/planner.hpp:49] compiler: total buffer size = 25690112/33918976, ratio = 0.757396
[nm_ort 7f4fbbd3f740 >DIAGNOSE< supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:644] == Beginning new subgraph ==
[nm_ort 7f4fbbd3f740 >WARN<  supported_subgraphs /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/supported/subgraphs.cc:652] Filtered subgraph was empty, ignoring subgraph.
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:122] Created 1 compiled subgraphs.
[nm_ort 7f4fbbd3f740 >DIAGNOSE< validate_minimum_supported_fraction /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/utility/graph_util.cc:321] == NM Execution Provider supports 70% of the network
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:129] == End NMExecutionProvider::GetCapability ==
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:140] == NMExecutionProvider::Compile ==
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:157] Graph #0: 1 inputs and 1 outputs
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:276] == End NMExecutionProvider::Compile ==
Generating 1 random inputs
  -- 1 random input of shape = [1, 1, 224, 224]
[     INFO         execute.py: 242 -   nm_exec_test_iters() ] Starting tests
[     INFO     neuralmagic.py: 121 -  neuralmagic_execute() ] Executing TEST_1
[     INFO     neuralmagic.py: 124 -  neuralmagic_execute() ]  [1] input_data.shape = (1, 1, 224, 224)
[     INFO     neuralmagic.py: 126 -  neuralmagic_execute() ]  -- START
[nm_ort 7f4fbbd3f740 >DIAGNOSE< operator() /home/jdoe/code/nyann/src/onnxruntime_neuralmagic/nm_execution_provider.cc:265] ORT NM EP compute_func: 6.478 ms
[     INFO     neuralmagic.py: 130 -  neuralmagic_execute() ]  -- FINISH
[     INFO     neuralmagic.py: 132 -  neuralmagic_execute() ]  [output] output_data.shape = (1, 1, 672, 672)
```
