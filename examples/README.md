# Examples

This directory contains actively maintained sample scripts to illustrate how to make use of the DeepSparse Engine. 

For instructions on how to run each example, either check the script header or run the example with `-h`.

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
| [Benchmark ONNX Model](https://github.com/neuralmagic/deepsparse/tree/main/examples/benchmark/run_benchmark.py)  | Script to benchmark performance between DeepSparse Engine and ONNXRuntime  |
| [Check Correctness ONNX Model](https://github.com/neuralmagic/deepsparse/tree/main/examples/benchmark/check_correctness.py)  | Script to check correctness between DeepSparse Engine and ONNXRuntime  |
| [Classification](https://github.com/neuralmagic/deepsparse/tree/main/examples/classification/classification.py)  | How to use classification models from SparseZoo and inference with the DeepSparse Engine  |
| [Detection](https://github.com/neuralmagic/deepsparse/tree/main/examples/detection/detection.py)  | How to use object detection models from SparseZoo and inference with the DeepSparse Engine  |
