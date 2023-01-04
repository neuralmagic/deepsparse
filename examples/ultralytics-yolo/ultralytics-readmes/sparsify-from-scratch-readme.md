# Sparsifying YOLOv5 with SparseML

This guide explains how to apply pruning and quantization to create an inference-optimized sparse 
version of YOLOv5.

SparseML is an open-source library which enables you to easily apply pruning and quantization algorithms to 
your models. Ultralytics is integrated with SparseML, allowing you to apply the algorithms from 
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

Pruning and Quantization can be applied with minimal accuracy loss when performed in a training-aware manner. This allows the model to slowly adjust to the new optimization space as the pathways are removed or become less precise. 

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
    
## Creating Sparsification Recipes

SparseML uses YAML-files called Recipes to encode the hyperparameters of the sparsification algorithms. The rest of the SparseML system parses the Recipes to setup GMP and QAT.

The easiest way to collect a Recipe for usage with YOLOv5 is downloading from the open-source SparseZoo model repository. [SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1) has a Sparsification Recipe available for each version of YOLOv5 and YOLOv5p6. 

<details>
    <summary>Click for an example of a simple Sparsification Recipe</b></summary>
    </br>

```yaml
# recipe.yaml
    
modifiers:
    - !GMPruningModifier
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

This recipe instructs SparseML to do the following:
- First, apply the GMP algorithm is to all layers, starting from an initial sparsity of 5% and gradually increasing to 80% over 30 epochs, as indicated by the `GMPruningModifier` element. 
- Second, fine tune for 20 epochs with at 80% sparsity.
- Finally, apply the QAT algorithm to all layers over the last 5 epochs, as indicated by `QuantizationModifier`.

Note that this Recipe is a simple example. You can find a state-of-the-art Sparsification Recipe for YOLOv5s in [SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-s%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned_quant-aggressive_94).

</details>


## Applying Sparsification Recipes to YOLOv5

Once you have a Recipe, you can kick off the sparsification process with the `train.py` script.

This example uses a Sparsification Recipe for YOLOv5s from SparseZoo, identified by the following stub:

```bash
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none
```

Run the following to download the Recipe from SparseZoo and sparsify YOLOv5s:

```bash
python train.py \
    --weights yolov5s.pt \
    --sparsification-recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/prunedXXX_quant-none \
    --data coco.yaml
```

Once the training completes, you will have a pruned-quantized version of YOLOv5s! 

The majority of layers are pruned to [**XX**]% and the weights have been quantized to INT8. 
On our training run, final accuracy is [**XX**] mAP@0.5, an [**XX**]% recovery against the dense baseline.

> Note: this example uses a SparseZoo stub, but you can also pass a local path to a `sparsification-recipe`.

## Exporting to ONNX

Many inference runtimes accept ONNX as the input format.

SparseML provides an export script that you can use to create a `model.onnx` version of your
trained model. The export process ensures that the quantized and pruned models are 
corrected and folded properly.

```
python3 export.py --weights runs/train/exp/weights/last.pt 
```

You have successfully created and exported a inference-optimized sparse version of YOLOv5.

Be sure to deploy with a sparsity-aware inference runtime to gain a performance speedup!
