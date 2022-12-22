# Sparsifying YOLOv5 with SparseML

This guide explains how to apply pruning and quantization to create an inference-optimized sparse 
version of YOLOv5.

SparseML is an open-source library which enables you to apply pruning and quantization algorithms to 
create sparse models. Ultralytics is integrated with SparseML, enabling you to apply sparsity from 
within the YOLOv5 repo.

## Installation

Clone the repo and install the requirements.

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Sparsity Overview

Introducing Sparsity to YOLOv5 can improve inference performance, especially when paired with 
an inference runtime that implements sparsity-aware optimizations.

SparseML uses two techniques to create sparse models:
- **Pruning** systematically removes redundant weights from a network
- **Quantization** reduces model precision by converting weights from `FP32` to `INT8`

Pruning and Quantization can be applied with minimal accuracy loss when performed in a training-aware manner with 
access to training data. This allows the model to slowly adjust to the new optimization space as the pathways are removed or become less precise. 

See below for more details on the key algorithms:

<details>
    <summary><b>Pruning: GMP</b></summary>
    <br>
   
Gradual magnitude pruning or GMP is the best algorithm for pruning. With it, 
the least impactful weights are iteratively removed over several epochs up to a specified level of sparsity. 
The remaining non-zero weights are then fine-tuned to the objective function. This iterative process enables 
the model to slowly adjust to a new optimization space after pathways are removed before pruning again.

</details>
        
<details>
    <summary><b>Quantization: QAT</b></summary>
    <br>

Quantization aware training or QAT is the best algorithm for quantization. With it, fake quantization 
operators are injected into the graph before quantizable nodes for activations, and weights 
are wrapped with fake quantization operators. The fake quantization operators interpolate 
the weights and activations down to `INT8` on the forward pass but enable a full update of 
the weights at `FP32` on the backward pass. This allows the model to adapt to the loss of 
information from quantization on the forward pass. 
    
</details>
    
## Creating SparseML Recipes

SparseML uses YAML-files called Recipes to encode the hyperparameters of the sparsification algorithms. The rest of the SparseML system parses the Recipes to setup GMP and QAT.

The easiest way to create a Recipe for usage with SparseML is downloading a pre-made Recipe
from the open-source SparseZoo model repository. [SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1) has a sparsification recipe available for each version of YOLOv5 and YOLOv5p6. 

>:rotating_light: **Pro-Tip:** Consider using [Sparse Transfer Learning **UPDATE LINK**](Ultralytics-STL-README.md). 
>It is an easier way to create a sparse model trained on your data.

We will explain the `Modifiers` used in the Recipes for **GMP** and **QAT**. 

<details>
    <summary><b>GMP Modifiers</b></summary>
    <br>

An example `recipe.yaml` file for GMP is the following:

```yaml
# gmp-recipe.yaml
   
modifiers:
    - !GlobalMagnitudePruningModifier
        init_sparsity: 0.05
        final_sparsity: 0.8
        start_epoch: 0.0
        end_epoch: 30.0
        update_frequency: 1.0
        params: __ALL_PRUNABLE__

    - !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 0.05

    - !LearningRateFunctionModifier
        start_epoch: 30.0
        end_epoch: 50.0
        lr_func: cosine
        init_lr: 0.05
        final_lr: 0.001

    - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 50.0
```

Each `Modifier` encodes a hyperparameter of the **GMP** algorithm:
  - `GlobalMagnitudePruningModifier` applies gradual magnitude pruning globally across all the prunable parameters/weights in a model. It
  starts at 5% sparsity at epoch 0 and gradually ramps up to 80% sparsity at epoch 30, pruning at the start of each epoch.
  - `SetLearningRateModifier` sets the pruning LR to 0.05 (the midpoint between the original 0.1 and 0.001 LRs used to train YOLO).
  - `LearningRateFunctionModifier` cycles the LR from 0.5 to 0.001 with a cosine curve (0.001 was the final original training LR).
  - `EpochRangeModifier` expands the training time to continue finetuning for an additional `20` epochs after pruning has ended.

30 pruning epochs and 20 finetuning epochs were chosen based on a 50 epoch training schedule - be sure to adjust based on the number of epochs as needed.

</details>

<details>
    <summary><b>QAT Modifiers</b></summary>
    <br>
    
An example `recipe.yaml` file for QAT is the following:

```yaml
# qat-recipe.yaml
    
modifiers:
    !QuantizationModifier
        start_epoch: 0.0
        submodules: ['model']
        freeze_bn_stats_epoch: 3.0

    !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 10e-6

    !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 5.0
```

Each `Modifier` encodes a hyperparameter of the **QAT** algorithm:  

  - The `QuantizationModifier` applies QAT to all quantizable modules under the `model` scope.
Note the `model` is used here as a general placeholder; to determine the name of the root module for your model, print out the root module and use that root name.
  - The `QuantizationModifier` starts at epoch 0 and freezes batch normalization statistics at the start of epoch 3.
  - The `SetLearningRateModifier` sets the quantization LR to 10e-6 (0.01 times the example final LR of 0.001).
  - The `EpochRangeModifier` sets the training time to continue training for the desired 5 epochs.

</details>

<details>
    <summary><b>Compound Sparsity: GMP + QAT</b></summary>
    </br>
    
Pruning and quantization can be applied together. When run in a sparsity-aware runtime, the speedup
from pruning and quantization amplify eachother. Here's what a Recipe might look like with both GMP and QAT:

```yaml
# recipe.yaml
    
modifiers:
    - !GlobalMagnitudePruningModifier
        init_sparsity: 0.05
        final_sparsity: 0.8
        start_epoch: 0.0
        end_epoch: 30.0
        update_frequency: 1.0
        params: __ALL_PRUNABLE__

    - !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 0.05

    - !LearningRateFunctionModifier
        start_epoch: 30.0
        end_epoch: 50.0
        lr_func: cosine
        init_lr: 0.05
        final_lr: 0.001

    - !QuantizationModifier
        start_epoch: 50.0
        freeze_bn_stats_epoch: 53.0

    - !SetLearningRateModifier
        start_epoch: 50.0
        learning_rate: 10e-6

    - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 55.0
```
    
</details>

Checkout SparseML's [Recipe User Guide](https://docs.neuralmagic.com/user-guide/recipes/creating) 
for more details on creating recipes.

## Applying Sparsification Recipes to YOLOv5

Once you have created a Recipe or identifed a Recipe in the SparseZoo, you can use the SparseML-YOLOv5 integration 
to kick off the sparsification process with the `train.py` script.

We will use YOLOv5s as the starting point and the pre-made sparsification recipe from SparseZoo, identified by the following stub:

```bash
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none
```

The following CLI command downloads the sparsification recipe from the SparseZoo and 
kicks off the sparsification process, fine-tuning onto the COCO dataset.

```bash
python train.py \
    --weights yolov5s.pt \
    --sparsification-recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none \
    --data coco.yaml
```

Once the training completes, you will have a pruned-quantized version of YOLOv5s! The majority of layers are pruned to [**XX**]% and the weights have been quantized to INT8. On our training run, final accuracy is [**XX**] mAP@0.5, an [**XX**]% recovery against the dense baseline.

> Note: this example uses a SparseZoo stub, but you can also pass a local path to a `sparsification-recipe`.

## Exporting to ONNX

Many inference runtimes accept ONNX as the input format.

SparseML provides an export script that you can use to create a `model.onnx` version of your
trained model. The export process is modified such that the quantized and pruned models are 
corrected and folded properly. Point the `--weights` argument points to your trained model.

```
python3 export.py --weights runs/train/exp/weights/last.pt 
```

You have successfully created and exported a inference-optimized sparse version of YOLOv5.

Be sure to deploy with a sparsity-aware inference runtime to gain a performance speedup!
