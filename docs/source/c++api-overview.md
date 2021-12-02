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

# C++ API Overview

You can use a C++ API in `libdeepsparse.so` as the interface between your application and the [Neural Magic DeepSparse Engine](https://docs.neuralmagic.com/deepsparse/). You would start with a model that has been exported in ONNX format, client code on your server. Your application will write data to input tensors, execute the DeepSparse Engine for inference results, and read output tensors for result data.

A simple demo with code is provided to invoke the DeepSparse Engine using the C++ API. Once you have installed the DeepSparse Engine, you will be ready to use the C++ API and take advantage of the library `libdeepsparse`. With `libdeepsparse`, you can run the demo by building and running `demo.bin`.

You can find more information about the engine at [https://docs.neuralmagic.com/deepsparse/](https://docs.neuralmagic.com/deepsparse/).


## Prerequisites

The following is required to build and run the demo.

**OS** - Our engine binary is manylinux2014-compatible and built on CentOS 7.  Linux distributions such as Ubuntu 20.04, which are compatible with manylinux2014, should support it.


**Hardware** - The utility arch.bin is included in deepsparse_api_demo.tar.gz. Run `arch.bin` to check hardware support. The “isa” field in the output states whether you are running on a machine that supports the avx512 or avx2 instruction set.

```
./arch.bin
```

**Installed Tools** - These tools are assumed to be installed to build the demo:

* clang++ 11  or g++ 10
* C++17 standard libraries


## Demo: Obtaining, Building, and Running

To download a demo file, use the curl command:


```
curl https://github.com/neuralmagic/deepsparse/releases/download/v0.8.0/deepsparse_api_demo.tar.gz --output deepsparse_api_demo.tar.gz
```
or visit our [DeepSparse Engine Release page](https://github.com/neuralmagic/deepsparse/releases/tag/v0.8.0) and download `deepsparse_api_demo.tar.gz` from there.


Once you obtain the file deepsparse-api.tar.gz, follow these steps to unpack and build the demo:
<YOUR ARCH> should be either avx2 or avx512 based on the result of arch.bin.

```
tar xzvf deepsparse_api_demo.tar.gz
cd <YOUR ARCH>_deepsarse_api_demo
make
./bin/<YOUR ARCH>/demo.bin ./data/model.onnx
```

**Note**.  The makefile defaults to avx512 support. For tar files and machines with avx2 instruction set, to build, you must set the ARCH flag on the command line when you make the demo.

```
make ARCH=avx2
```

## C++ API

This document discusses the high-level overview of the API. For the exact signatures and classes of the API, review the header files under

```
deepsparse_api_demo/include/libdeepsparse/
```

The API consists of five C++ header files:

```
compiler.hpp
config.hpp
dimensions.hpp
tensor.hpp
engine.hpp
```

### Compiler

Helper header to export the API to a shared object.

### Config

This file contains a structure that is used in the call to create the engine. The **engine_config_t** fields are:

**model_file_path** - This should be a file path to the model in the ONNX file format.  See [DeepSparse Engine documentation](https://docs.neuralmagic.com/deepsparse/) on proper ONNX model files.

**batch_size** - The batch size refers to the process of concatenating input and output tensors into a contiguous batched tensor. See [DeepSparse Engine documentation](https://docs.neuralmagic.com/deepsparse/) about the performance trade-offs of batching.

**num_threads** - The number of worker threads for the engine to use while executing a model.  If left as 0, the engine will select a default number of worker threads.


### Dimensions

Review the [DeepSparse Engine documentation](https://docs.neuralmagic.com/deepsparse/) about expected input and output tensors. The **dimensions_t** object describes the extent or count of elements along each dimension of a tensor_t.


### Tensor

A tensor is an n-dimensional array of data elements and metadata. An element is a concrete value of a supported primitive type (for example, an element of type float or uint8).

**tensor_t** - members.

**element_type()** - Return an enumeration value of the concrete type of a tensor element.

**dims()** - Return a dimension_t that specifies the extents of the tensor.

**data()** - Pointer to the first element of the tensor data memory.

The data that tensor_t points to may have either of two different lifetime models for the memory.

1. The tensor owns the data memory.
2. The tensor aliases a pointer to the memory location. The lifetime of the data memory is delegated to a lambda passed to the tensor. For externally-owned data pointers, the lambda can be a no-op.

_Code Example: Memory lifetime allocated and owned by tensor_t_


```
#include "deepsparse/engine.hpp"

void my_engine_inference()
{
    // ...
    {
        // create a float tensor with y and x 10 by 10
        auto t = deepsparse::create_tensor(
            deepsparse::element_type_t::float32,
            deepsparse::dimensions_t{1, 1, 1, 10, 10});

        float* p_raw = t.data<float>(); // 100 floats
        // read and write values to p_raw and send tensor to the engine.


    }
    // data memory of t deallocated when t goes out of scope
    // ...
}
```

_Code Example: Tensor data memory lifetime is delegated to lambda_

```
#include <cstdlib>
#include "deepsparse/engine.hpp"

void my_engine_inference()
{
    // tensor data memory MUST aligned.
    // dims must match total number of elements below
    float* p_raw = static_cast<float*>(aligned_alloc(
            deepsparse::minimum_required_alignment(),
            sizeof(float) * 100));
    {

        // policy - tensor is just an alias to memory
        auto alias_dealloc = [](void* p) {};

        // policy - tensor destructor calls dealloc
        auto owned_dealloc = [](void* p) {
            free(p);
            // or cast p to pointer to well known type
            // and manually call your own delete
        };

        // create a alias float tensor with y and x 10 by 10
        auto t = deepsparse::tensor_t(
            deepsparse::element_type_t::float32,
            deepsparse::dimensions_t{1, 1, 1, 10, 10},
            p_raw,
            alias_dealloc // lambda invoked in object destructor
        );

        // read and write p_raw and send to the engine.
    }

}
```

### Engine

The engine API is the primary interface for external code to load and run the [Neural Magic DeepSparse Engine](https://docs.neuralmagic.com/deepsparse/).

**engine_t()** - Construct an instance of the engine with the configuration struct. The config file path to the model is used to load and compile the model during the constructor. On error, exceptions will be thrown to the calling code.

**execute()** - This is the primary method to run the inference specified in the ONNX model.  The engine executes the model with the input tensors and returns output tensors.

**Input and output getters** - these methods are used to get metadata on the input and output tensors of the loaded model.


#### Utility Functions

**generate_random_inputs()** - Once the engine is instantiated and has a model loaded, this function can use the model’s definition of input tensors to generate a set of tensors with random values with the correct type and shape. Random input can be used to test the engine or its performance.

**load_inputs()** - Creates a collection of tensor_t from the model input files. The input files are expected to be in the NumPy array format. See the [DeepSparse Engine documentation](https://docs.neuralmagic.com/deepsparse/) for more on input file support.

**max_diff()** - Returns the largest absolute elementwise difference between two tensors. This function is used to compare the output result with the expected output when testing the engine.

**allclose()** - Returns true if two tensors are less elementwise different than the specified absolute and relative tolerances.


### User Calling Code

The expected workflow is that the calling code will create one engine per model. The model path will specify an ONNX file. During creation, the engine will load and compile the model from the file.

The input tensors will be created by the calling code. The tensor dimensions and types must match the corresponding dimensions and types of the model inputs. And, the number of tensors must be the same as the number of model inputs.

Call engine execute() with the collection of input tensors. During the execute() call, the engine will do the inference over the ONNX model with the input tensors and return the output tensors.

The output tensors’ data members can then be read to extract values and results.

For a detailed example of creating, loading, and running an engine, see the code in `deepsparse_api_demo/src/demo.cpp`
