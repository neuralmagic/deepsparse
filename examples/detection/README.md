# Object Detection Example

This directory holds an example for downloading an object detection model from SparseZoo with real data and using the DeepSparse Engine for inference and benchmarking.

## Installation

Install DeepSparse with `pip install deepsparse`.

## Notebook

There is a step-by-step [detection.ipynb notebook](https://github.com/neuralmagic/deepsparse/blob/main/notebooks/detection.ipynb) for this example.

## Execution

Example command for running a `yolo_v3` model with batch size 8 and 4 cores:
```bash
python detection.py yolo_v3 --batch_size 8 --num_cores 4
```