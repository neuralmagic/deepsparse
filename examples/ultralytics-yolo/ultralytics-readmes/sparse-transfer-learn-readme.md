# Sparse Transfer Learning with SparseML

This guide explains how to fine-tune an inference-optimized sparse version of YOLOv5 onto a custom dataset. 

SparseML is an open-source library which enables you to apply pruning and quantization algorithms to create sparse models. Ultralytics is integrated with SparseML, enabling you to apply Sparse Transfer Learning from within the YOLOv5 repo.

## Installation

Clone the repo and install the requirements.

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Sparse Transfer Learning Overview

Sparse Transfer Learning is the easiest way to create a sparse version of YOLOv5 trained on custom data. Using a sparse model can improve inference performance, especially when paired 
with an inference runtime that implements sparsity-aware optimizations. 
                                                                                                       
Similiar to typical fine-tuning, Sparse Transfer Learning starts with a checkpoint trained on a large dataset and then fine-tunes the weights onto a smaller downstream dataset. However, with Sparse Transfer Learning, the training starts from a sparse checkpoint and the sparsity structure is maintained as the fine-tuning occurs. The end result is a sparse model trained on the downstream dataset!
                                                                                                            
> **Clarification:** When we say sparse models, we are describing sparsity in the **weights** of the model. 

There are four simple steps to Sparse Transfer Learning:
1. Select a Pre-Sparsified Model as the Starting Point
2. Create Downstream Dataset
3. Run Sparse Transfer Learning Algorithm
4. Export to ONNX

## 1. Select a Pre-Sparsified Model

[SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1)
is an open-source repository of state-of-the-art pre-sparsified models, including a sparse version of each YOLOv5 and YOLOv5p6 model. SparseML is 
integrated with SparseZoo, so you can easily use SparseZoo checkpoints in the fine-tuning process.

In this example, we will use a pruned-quantized **YOLOv5s** from the SparseZoo. The layers are pruned to **[XXX%]** and the weights are quantized to INT8. It achieves and mAP@0.5 of **[XXX%]**, which is **[XXX%]** of the accuracy versus the dense baseline on COCO.

It is identifed by the following SparseZoo stub, which SparseML uses to download the model:

```
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none
```

## 2. Create Dataset

The [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-dataset) 
tutorial has a detailed explanation on creating a custom dataset. In this example we will use the VOC 
dataset, for which there is a configuration file available off the shelf (`VOC.yaml`).

## 3. Run Sparse Transfer Learning

You can run Sparse Transfer Learning with the `train.py` script. The only change relative to typical training is that you must point to a sparse model as the starting weights and provide a Sparse Transfer Learning Recipe.

### Sparse Transfer Learning Recipes

SparseML uses YAML-files called Recipes to encode the the hyperparameters of sparsity-related algorithms. Sparse Transfer Learning Recipes instruct SparseML to maintain the sparsity structure of the networks while fine-tuning the model.

<details>
  
  <summary>Click to learn more about Recipes</summary>
  <br>
  
Recipes are YAML files that encode sparsity-related hyperparameters. Modifiers within a Recipe instruct SparseML which 
algorithms to apply during the learning process to induce or preserve sparsity within a network.

For Sparse Transfer Learning, the key Modifiers are:
- `ConstantPruningModifier` which instructs SparseML to maintain the starting sparsity level as it fine-tunes
- `QuantizationModifier` which instructs SparseML to quantize the model 

For example, the Sparse Transfer Learning Recipe for YOLOv5s looks like the following:
  
```yaml

# General Epoch/LR Hyperparams
num_epochs: 52
init_lr: 0.0032
final_lr: 0.000384
warmup_epochs: 2
weights_warmup_lr: 0
biases_warmup_lr: 0.05
quantization_lr: 0.000002

# Quantization Params
quantization_start_epoch: 50

# modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)
    
  - !LearningRateFunctionModifier
    start_epoch: eval(warmup_epochs)
    end_epoch: eval(num_epochs)
    lr_func: cosine
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(warmup_epochs)
    lr_func: linear
    init_lr: eval(weights_warmup_lr)
    final_lr: eval(init_lr)
    param_groups: [0, 1]
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(warmup_epochs)
    lr_func: linear
    init_lr: eval(biases_warmup_lr)
    final_lr: eval(init_lr)
    param_groups: [2]
    
  - !SetLearningRateModifier
    start_epoch: eval(quantization_start_epoch)
    learning_rate: eval(quantization_lr)

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
    
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(quantization_start_epoch)
    submodules: [ 'model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 'model.6', 'model.7', 'model.8', 'model.9', 'model.10', 'model.11', 'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22', 'model.23' ]
```

The `pruning_modifiers` and `quantization_modifiers` sections are where the magic happens. The recipe instructs SparseML to maintain sparsify for all prunable layers over every epoch (as indicated by the parameters of `ConstantPruningModifier`) and to apply quantization to 24 layers over the final 3 epochs (as indicated by the parameters of `QuantizationModifier`).
 
The `training_modifiers` section simply controls the learning rates during the pruning and quantization processes.
  
</details>

### Run The Algorithm

Kick off training with `train.py`, passing in the starting checkpoint, the sparsification recipe, and dataset config.

```bash
python3 train.py \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none?recipe_type=transfer_learn \
  --sparsification-recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none?recipe_type=transfer_learn \
  --data VOC.yaml
```

Once the training is finished, you will have a pruned-quantized YOLOv5s model trained on VOC!

> Note: the example uses SparseZoo stubs, but you can also pass a local path as the `weights` and `sparsification-recipe`.

## 4. Exporting to ONNX

Many inference runtimes accept ONNX as the input format.

SparseML provides a script that you can use to export to ONNX. SparseML's export process 
ensures that the quantized and pruned models properly translated to ONNX. Be sure the `--weights` argument points to your trained model.

```
python3 export.py --weights runs/train/exp/weights/last.pt 
```

You have successfully created and exported a inference-optimized sparse version of YOLOv5 trained on custom data! 

Be sure to deploy with a sparsity-aware inference runtime to gain a performance speedup!
