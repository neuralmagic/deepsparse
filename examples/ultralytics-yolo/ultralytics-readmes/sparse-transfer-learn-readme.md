# Sparse Transfer Learning with YOLOv5 :rocket:

:books: This guide explains how to fine-tune a sparse YOLOv5 :rocket: onto a custom dataset with SparseML.

## :arrow_heading_down: Installation

#### [UPDATE WITH NEW PATHWAY]

SparseML is an open-source library that includes tools to create sparse models. SparseML is integrated with Ultralytics, allowing you to easily apply Sparse Transfer Learning to YOLOv5 and YOLOv5p models.

Install SparseML with the following command. We recommend using a virtual enviornment.
```bash
pip install sparseml[torchvision]
```

## 💡 Conceptual Overview

Sparse Transfer Learning is the **easiest** way to create a sparse YOLOv5 model trained on custom data. 
                                                                                                        
Similiar to typical transfer learning you might be familiar with, Sparse Transfer Learning starts with a sparse model trained on a large dataset and fine-tunes the weights onto a smaller downstream dataset. However, with Sparse Transfer Learning, we maintain the sparsity structure of the starting model as the fine-tuning occurs. As such, the final trained model will have the same level of sparsity as the starting model.
                                                                                                            
>:rotating_light: **Clarification:** When we say sparse models, we are describing sparsity in the **weights** of the model. 
With proper pruning, you can set [XX]% of YOLOv5-l weights to 0 and retain [XX]% of the dense model's accuracy. 
See [Sparsifying YOLOv5](Ultralytics-Sparsify-README.md) for more details.

Sparse Transfer Learning enables you to train a sparse model on your dataset with just one command line call.

## :mag_right: How It Works

There are four simple steps to Sparse Transfer Learning with SparseML:
1. Select a Pre-Sparsified Model
2. Create Dataset
3. Run Sparse Transfer Learning Algorithm
4. Export to ONNX

## 1. Select a Pre-Sparsified Model

[SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1)
is an open-source repository of state-of-the-art pre-sparsified models. There is a sparse version of each
YOLOv5 and YOLOv5p model available as the starting point for Sparse Transfer Learning.

In this example, we will use **[xxx]**. The majority of layers are pruned between [xx%] and [xx%] and it 
achieves [xxx%] recovery of the performance for the dense baseline on COCO. 

It is identifed by the following SparseZoo stub (which SparseML uses to download the model):
```
# [XXX] add the stub
```

## 2. Create Dataset

The [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-dataset) tutorial has a detailed 
explanation on creating a custom dataset. SparseML is integrated with Ultralytics and accepts data in the same format. SparseML 
contains a config file with a download script for the VOC dataset, which will be used in this example.

## 3. Run Sparse Transfer Learning

SparseML is integrated with Ultralytics, so you can kick off a training run with a simple CLI command.

All you have to do is specify a dataset, a base pre-sparsified model, and a transfer learning recipe.

### :cook: Transfer Learning Recipes

SparseML uses **Recipes** to encode the the hyperparameters of the Sparse Transfer Learning algorithm. SparseZoo has pre-made Recipes for every version of YOLOv5 and YOLOv5p off-the-shelf. 
>:rotating_light: You should use the off-the-shelf recipes from SparseZoo (tweaking the number of epoch and learning rate if needed).

<details>
  
  <summary>Click to learn more</summary>
  <br>
  
You can see details on **Recipes** in the [Sparsifying YOLOv5 Tutorial **UPDATE LINK**](Ultralytics-Sparsify-README.md#cook-creating-sparseml-recipes) if interested. 

For Sparse Transfer Learning, the key **Modifiers** in the recipe are:
- `ConstantPruningModifier` which instructs SparseML to maintain the starting sparsity level as it fine-tunes
- `QuantizationModifier` which instructs SparseML to quantize the model 

For example, in the [XXX Transfer Learning recipe UPDATE LINK](link), the following lines are included in the recipe:

```yaml
pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
    
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(quantization_start_epoch)
    submodules: [ 'model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 'model.6', 'model.7', 'model.8', 'model.9', 'model.10', 'model.11', 'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22', 'model.23' ]
```
</details>

### 🏋️ Run The Algorithm

We will Sparse Transfer Learn **pruned-quantized XXX** onto the VOC dataset. 

The `xxx` CLI command downloads the model and VOC dataset (using the download script from `VOC.yaml`) and kicks off the training process using the pre-made recipe from SparseZoo.

```bash
# update with the new pathway
```
  - `--data` is the config file for the dataset
  - `--weights` identifies the base pre-sparsfied model for the transfer learning. It can be a SparseZoo stub or a path to a local model
  - `--recipe` identifies the transfer learning recipe. It can be SparseZoo stub or a path to a local recipe

Once the training is finished, you will have a pruned-quantized [XXX] model trained on VOC!

> Note: the example passes SparseZoo stubs as the `weights` and `recipe`, but you can also pass a local path to a model / recipe.

## 4. Exporting to ONNX

Many inference runtimes accept ONNX as the input format.

SparseML provides a script that you can use to export to ONNX. SparseML's export process 
ensures that the quantized and pruned models properly translated to ONNX. Be sure the `--weights` argument points to your trained model.

```
sparseml.yolov5.export_onnx \
   --weights path/to/weights.pt \
   --dynamic
```

You have successfully created and exported a inference-optimized sparse version of YOLOv5 trained on custom data! Be sure to deploy with a sparsity-aware runtime to gain a performance speedup!
