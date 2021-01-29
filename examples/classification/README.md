# Image Classification Example

This directory holds an example for downloading an image classification model from SparseZoo with real data and using the DeepSparse Engine for inference and benchmarking.

## Installation

Install DeepSparse with `pip install deepsparse`.

## Notebook

There is a step-by-step [classification.ipynb notebook](https://github.com/neuralmagic/deepsparse/blob/main/notebooks/classification.ipynb) for this example.

## Execution

Example command for running a `mobilenet_v2` model with batch size 8 and 4 cores used:
```bash
python classification.py mobilenet_v2 --batch_size 8 --num_cores 4
```

Run with the `-h` flag to see all available models.