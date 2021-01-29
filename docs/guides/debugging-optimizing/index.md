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

# Debugging and Optimizing Performance

- [Using the numactl Utility to Control Resource Utilization with the DeepSparse Engine](./numactl-utility.md#using-the-numactl-utility-to-control-resource-utilization-with-the-deepsparse-engine)
  - [DeepSparse Engine and Thread Pinning](./numactl-utility.md#deepsparse-Engine-and-thread-pinning)
  - [Additional Notes](./numactl-utility.md#additional-notes)

- [Logging Guidance for Diagnostics and Debugging](diagnostics-debugging.md#logging-guidance-for-diagnostics-and-debugging)
  - [Performance Tuning](./diagnostics-debugging.md#performance-tuning)
  - [Enabling Logs and Controlling the Amount of Logs Produced by the DeepSparse Engine](./diagnostics-debugging.md#enabling-logs-and-controlling-the-amount-of-logs-produced-by-the-deepsparse-engine)
  - [Parsing an Example Log](./diagnostics-debugging.md#parsing-an-example-log)
    - [Viewing the Whole Graph](./diagnostics-debugging.md#finding-supported-nodes-for-our-optimized-nmie)
    - [Finding Supported Nodes for Our Optimized Engine](./diagnostics-debugging.md#finding-supported-nodes-for-our-optimized-engine)
    - [Compiling Each Subgraph](./diagnostics-debugging.md#compiling-each-subgraph)
    - [Determining the Number of Cores and Batch Size](./diagnostics-debugging.md#determining-the-number-of-cores-and-batch-size)
    - [Obtaining Subgraph Statistics](./diagnostics-debugging.md#obtaining-subgraph-statistics)
    - [Viewing Runtime Execution Times](./diagnostics-debugging.md#viewing-runtime-execution-times)
  - [Example Log, Verbose Level = diagnose](example-log.md)
