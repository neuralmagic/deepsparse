# Sparse Transfer Learning with YOLOv5

This guide explains how to fine-tune a sparse version of YOLOv5 onto a custom dataset.

SparseML is an open-source library which enables you to apply pruning and quantization algorithms to create
inference-optimized sparse model. Ultralytics is integrated with SparseML, enabling you to apply sparsity from 
within the YOLOv5 repo.

## Installation

Clone the repo and install the requirements.

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Sparse Transfer Learning Overview

Sparse Transfer Learning is the easiest way to create an inference-optimized sparse YOLOv5 model trained on custom data. 
                                                                                                        
Similiar to typical fine-tuning, Sparse Transfer Learning starts with a checkpoint trained on a large dataset and then fine-tunes the weights onto a smaller downstream dataset. However, with Sparse Transfer Learning, the training starts from a sparse checkpoint and the sparsity structure is maintained as the fine-tuning occurs. The end result is a sparse model trained on the downstream dataset!
                                                                                                            
>:rotating_light: **Clarification:** When we say sparse models, we are describing sparsity in the **weights** of the model. 

There are four simple steps to Sparse Transfer Learning:
1. Select a Pre-Sparsified Model as the Starting Point
2. Create Downstream Dataset
3. Run Sparse Transfer Learning Algorithm
4. Export to ONNX

## 1. Select a Pre-Sparsified Model

[SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1)
is an open-source repository of state-of-the-art pre-sparsified models, including a sparse version of each YOLOv5 and YOLOv5p model. SparseML is 
integrated with SparseZoo, so you can easily use SparseZoo checkpoints in the fine-tuning process.

In this example, we will use a pruned-quantized **YOLOv5s** from the SparseZoo. The layers are pruned to **[XXX%]** and the weights are quantized to INT8. The model achieves and mAP@0.5 of **[XXX%]**, which is **[XXX%]** recovery of the accuracy versus the dense baseline on COCO.

It is identifed by the following SparseZoo stub, which SparseML uses to download the model:

```
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none
```

## 2. Create Dataset

The [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-dataset) 
tutorial has a detailed explanation on creating a custom dataset. In this example we will use the VOC 
dataset, for which there is a configuration file available off the shelf (`voc.yaml`).

## 3. Run Sparse Transfer Learning

Ultralytics is integrated with SparseML, so you kick off the Sparse Transfer Learning with the`train.py` command. The only change is that you must point to a sparse model as the starting point and provide a Sparse Transfer Learning Recipe.

### Sparse Transfer Learning Recipes

SparseML uses Recipes to encode the the hyperparameters of sparsity-related algorithms. Sparse Transfer Learning Recipes instruct SparseML to maintain the sparsity structure of the networks while fine-tuning the model. A Sparse Transfer Learning Recipe is available for each version of YOLOv5 and YOLOv5p in the SparseZoo.

<details>
  
  <summary>Click to learn more about Recipes</summary>
  <br>
  
Recipes are YAML files that encode sparsity-related hyperparameters. Modifiers within the Recipe instruct SparseML which 
algorithms to apply during the learning process to induce or preserve sparsity within a network.

For Sparse Transfer Learning, the key Modifiers in the recipe are:
- `ConstantPruningModifier` which instructs SparseML to maintain the starting sparsity level as it fine-tunes
- `QuantizationModifier` which instructs SparseML to quantize the model 

For example, in the sparse transfer learning recipe for YOLOv5s looks like the following:
  
```yaml
**ADD RECIPE**
```
  
You can see that for the first XXX epochs, SparseML will fine-tune while preserving sparsity (as indicated by the parameters of `ConstantPruningModifier`) for the and then quantize the model over the final XXX epochs (as indicated by the parameters of `QuantizationModifier`).
  
</details>

### Run The Algorithm

Kick off training with `train.py`, passing in the starting checkpoint, the sparsification recipe, and dataset config. The fine-tuning
will occur as usual, but the sparsity structure of the network will be maintained.

```bash
python3 train.py \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none?recipe_type=transfer_learn \
  --sparsification-recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none?recipe_type=transfer_learn \
  --data voc.yaml
```

Once the training is finished, you will have a pruned-quantized YOLOv5s model trained on VOC!

> Note: the example passes SparseZoo stubs as the `weights` and `recipe`, but you can also pass a local path to a model / recipe.

## 4. Exporting to ONNX

Many inference runtimes accept ONNX as the input format.

SparseML provides a script that you can use to export to ONNX. SparseML's export process 
ensures that the quantized and pruned models properly translated to ONNX. Be sure the `--weights` argument points to your trained model.

```
python3 export.py --weights runs/train/exp/weights/last.pt 
```

You have successfully created and exported a inference-optimized sparse version of YOLOv5 trained on custom data! 

Be sure to deploy with a sparsity-aware inference runtime to gain a performance speedup!
