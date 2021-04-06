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

# Using the numactl Utility to Control Resource Utilization with the DeepSparse Engine

The DeepSparse Engine achieves better performance on multiple-socket systems as well as with hyperthreading disabled; models with larger batch sizes are likely to see an improvement. One standard way of controlling compute/memory resources when running processes is to use the **numactl** utility. **numactl** can be used when multiple processes need to run on the same hardware but require their own CPU/memory resources to run optimally.

To run the DeepSparse Engine on a single socket (N) of a multi-socket system, you would start the DeepSparse Engine using **numactl**. For example:

```bash
    numactl --cpunodebind N <deepsparseengine-process>
```

To run the DeepSparse Engine on multiple sockets (N,M), run:

```bash
    numactl --cpunodebind N,M <deepsparseengine-process>
```

It is advised to also allocate memory from the same socket on which the engine is running. So, `--membind` or `--preferred` should be used when using `--cpunodebind.` For example:

```bash
    numactl --cpunodebind N --preferred N <deepsparseengine-process>
    or
    numactl --cpunodebind N --membind N <deepsparseengine-process>
```

The difference between `--membind` and `--preferred` is that `--preferred` allows memory from other sockets to be allocated if the current socket is out of memory.  `--membind` does not allow memory to be allocated outside the specified socket.

For more fine-grained control, **numactl** can be used to bind the process running the DeepSparse Engine to a set of specific CPUs using `--physcpubind`. CPUs are numbered from 0-N, where N is the maximum number of logical cores available on the system. On systems with hyper-threading (or SMT), there may be more than one logical thread per physical CPU. Usually, the logical CPUs/threads are numbered after all the physical CPUs/threads. For example, in a system with two threads per CPU and N physical CPUs, the threads for a particular CPU (K) will be K and K+N for all 0&lt;=K&lt;N. The DeepSparse Engine currently works best with hyper-threading/SMT disabled, so only one set of threads should be selected using **numactl**, i.e., 0 through (N-1) or N through (N-1).

Similarly, for a multi-socket system with N sockets and C physical CPUs per socket, the CPUs located on a single socket will range from K*C to ((K+1)*C)-1 where 0&lt;=K&lt;N. For multi-socket, multi-thread systems, the logical threads are separated by N*C. For example, for a two socket, two thread per CPU system with 8 cores per CPU, the logical threads for socket 0 would be numbered 0-7 and 16-23, and the threads for socket 1 would be numbered 8-15 and 24-31.

Given the architecture above, to run the DeepSparse Engine on the first four CPUs on the second socket, you would use the following:

```bash
    numactl --physcpubind 8-11 --preferred 1 <deepsparseengine-process>
```

Appending `--preferred 1` is needed here since the DeepSparse Engine is being bound to CPUs on the second socket.

Note: When running on multiple sockets using a batch size that is evenly divisible by the number of sockets will yield the best performance.


## DeepSparse Engine and Thread Pinning

When using **numactl** to specify which CPUs/sockets the engine is allowed to run on, there is no restriction as to which CPU a particular computation thread is executed on. A single thread of computation may run on one or more CPUs during the course of execution. This is desirable if the system is being shared between multiple processes so that idle CPU threads are not prevented from doing other work.

However, the engine works best when threads are pinned (i.e., not allowed to migrate from one CPU to another). Thread pinning can be enabled using the `NM_BIND_THREADS_TO_CORES` environment variable. For example:

```bash
    NM_BIND_THREADS_TO_CORES=1 <deepsparseengine-process>
    or
    export NM_BIND_THREADS_TO_CORES=1 <deepsparseengine-process>
```

`NM_BIND_THREADS_TO_CORES` should be used with care since it forces the DeepSparse Engine to run on only the threads it has been allocated at startup. If any other process ends up running on the same threads, it could result in a major degradation of performance.

**Note:** The threads-to-cores mappings described above are specific to Intel only. AMD has a different mapping. For AMD, all the threads for a single core are consecutive, i.e., if each core has two threads and there are N cores, the threads for a particular core K are 2*K and 2*K+1.  The mapping of cores to sockets is also straightforward, for a N socket system with C cores per socket, the cores for a particular socket S are numbered S*C to ((S+1)*C)-1.

## Additional Notes

`numactl --hardware` </br>
Displays the inventory of available sockets/CPUs on a system.

`numactl --show` </br>
Displays the resources available to the current process.

For further details about these and other parameters, see the man page on **numactl**:

```bash
    man numactl
```
