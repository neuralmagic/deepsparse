# Debugging and Optimizing Performance

- [Using the numactl Utility to Control Resource Utilization with the Deep Sparse Engine](numactl-utility.md#using-the-numactl-utility-to-control-resource-utilization-with-the-deep-sparse-engine)
  - [Deep Sparse Engine and Thread Pinning](./numactl-utility.md#Deep-Sparse-Engine-and-thread-pinning)
  - [Additional Notes](./numactl-utility.md#additional-notes)

- [Logging Guidance for Diagnostics and Debugging](diagnostics-debugging.md#logging-guidance-for-diagnostics-and-debugging)
  - [Performance Tuning](./diagnostics-debugging.md#performance-tuning)
  - [Enabling Logs and Controlling the Amount of Logs Produced by the Deep Sparse Engine](./diagnostics-debugging.md#enabling-logs-and-controlling-the-amount-of-logs-produced-by-the-deep-sparse-engine)
  - [Parsing an Example Log](./diagnostics-debugging.md#parsing-an-example-log)
    - [Viewing the Whole Graph](./diagnostics-debugging.md#finding-supported-nodes-for-our-optimized-nmie)
    - [Finding Supported Nodes for Our Optimized Engine](./diagnostics-debugging.md#finding-supported-nodes-for-our-optimized-engine)
    - [Compiling Each Subgraph](./diagnostics-debugging.md#compiling-each-subgraph)
    - [Determining the Number of Cores and Batch Size](./diagnostics-debugging.md#determining-the-number-of-cores-and-batch-size)
    - [Obtaining Subgraph Statistics](./diagnostics-debugging.md#obtaining-subgraph-statistics)
    - [Viewing Runtime Execution Times](./diagnostics-debugging.md#viewing-runtime-execution-times)
  - [Example Log, Verbose Level = diagnose](example-log.md)
