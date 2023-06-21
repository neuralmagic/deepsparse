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

# Supported Hardware for DeepSparse

With support for AVX2, AVX-512, and VNNI instruction sets, DeepSparse is validated to work on x86 Intel (Haswell generation and later) and AMD (Zen 2 and later) CPUs running Linux.
Mac and Windows require running Linux in a Docker or virtual machine.

Here is a table detailing specific support for some algorithms over different microarchitectures:

| x86 Extension                                                                            | Microarchitectures                                                                                                                                                                                                                                                                                         | Kernel Sparsity | Sparse Quantization |
|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------:|:-------------------:|
| [AMD AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)      | [Zen 2,](https://en.wikipedia.org/wiki/Zen_2) [Zen 3](https://en.wikipedia.org/wiki/Zen_3)                                                                                                                                                                                                                 |    optimized    |    emulated    |
| [AMD AVX-512](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX-512) VNNI     | [Zen 4](https://en.wikipedia.org/wiki/Zen_4)                                                                                                                                                                                                                                                      |    optimized    |    optimized    |
| [Intel AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)    | [Haswell,](https://en.wikipedia.org/wiki/Haswell_%28microarchitecture%29) [Broadwell,](https://en.wikipedia.org/wiki/Broadwell_%28microarchitecture%29) and newer                                                                                                                                                  |    optimized    |    emulated    |
| [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512)                 | [Skylake](https://en.wikipedia.org/wiki/Skylake_%28microarchitecture%29), [Cannon Lake](https://en.wikipedia.org/wiki/Cannon_Lake_%28microarchitecture%29), and newer                                                                                                                                              |    optimized    |       emulated      |
| [Intel AVX-512](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512) VNNI (DL Boost) | [Cascade Lake](https://en.wikipedia.org/wiki/Cascade_Lake_%28microarchitecture%29), [Ice Lake](https://en.wikipedia.org/wiki/Ice_Lake_%28microprocessor%29), [Cooper Lake](https://en.wikipedia.org/wiki/Cooper_Lake_%28microarchitecture%29), [Tiger Lake](https://en.wikipedia.org/wiki/Tiger_Lake_%28microprocessor%29)  |    optimized    |      optimized      |
