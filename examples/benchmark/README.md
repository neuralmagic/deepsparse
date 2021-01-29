# Benchmarking and Correctness Examples

This directory holds examples for comparing inference on an ONNX model, both for performance and correctness.

## Installation

Install DeepSparse with `pip install deepsparse` and the additional external requirements with `pip install -r requirements.txt`.

## Execution

### Benchmark

`run_benchmark.py` is a script for benchmarking an ONNX model over random inputs and using both the DeepSparse Engine and ONNXRuntime, comparing results.

Example command for benchmarking a downloaded resnet50 model for batch size 8 and 4 cores, over 100 iterations:
```bash
python run_benchmark.py ~/Downloads/resnet50.onnx --batch_size 8 --num_cores 4 --num_iterations 100
```

### Check Correctness

`check_correctness.py` is a script for generating random input from an ONNX model and running the model both through the DeepSparse Engine and ONNXRuntime, comparing outputs to confirm they are the same.

Example command for checking a downloaded resnet50 model for batch size 8 and 4 cores:
```bash
python check_correctness.py resnet50.onnx --batch_size 8 --num_cores 4
```
