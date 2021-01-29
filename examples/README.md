# Examples

This directory contains self-documented examples to illustrate how to make use of the DeepSparse Engine. 

For instructions on how to run each example, either check the script header or run them with `-h`.

Open a Pull Request to [contribute](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md) your own examples.

## Important note

To run these scripts, you may need to install some packages for specific examples and ensure you have the correct release of `deepsparse` installed.

In a new virtual environment:
```base
pip install deepsparse
```
Then cd in the example folder of your choice and run
```bash
pip install -r requirements.txt
```

## Examples

| Notebook     |      Description      |
|----------|-------------|
| [Benchmark and ONNX Model Correctness](benchmark/)  | Comparing predictions and benchmark performance between DeepSparse Engine and ONNXRuntime  |
| [Classification](classification/)  | How to use classification models from SparseZoo to inference and benchmark with the DeepSparse Engine  |
| [Detection](detection/)  | How to use object detection models from SparseZoo to inference and benchmark with the DeepSparse Engine  |
| [Model Server](flask/)  | Simple model server and client example, showing how to use the DeepSparse Engine as an inference backend for a real-time inference server |
