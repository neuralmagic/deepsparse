# CLIP Inference Pipelines

DeepSparse allows inference on [CLIP](https://github.com/mlfoundations/open_clip) models.  

The CLIP integration currently supports the following task:
- **Zero-shot Image Classification** - Classifying images given possible classes
- **Caption Generation** - Generate a caption given an image

## Getting Started

Before you start your adventure with the DeepSparse Engine, make sure that your machine is compatible with our [hardware requirements](https://docs.neuralmagic.com/deepsparse/source/hardware.html).

### Installation
```pip install deepsparse[clip]```

### Model Format
By default, to deploy CLIP models using the DeepSparse Engine, it is required to supply the model in the ONNX format. This grants the engine the flexibility to serve any model in a framework-agnostic environment. To see examples of pulling CLIP models and exporting them to ONNX, please see the [sparseml documentation](https://github.com/neuralmagic/sparseml/tree/main/integrations/clip). 

For the Zero-shot image classification workflow, two ONNX models are required, a visual model for CLIP's visual branch, and a text model for CLIP's text branch. Both of these models can be produced through the sparseml integration linked above. For caption generation, specific models called CoCa models are required and instructions on how to export CoCa models are also provided in the sparseml documentation. The CoCa exporting pathway will generate one additional decoder model, along with the text and visual models.

### Deployment examples:
The following example uses pipelines to run the CLIP models for inference. For Zero-shot prediction, the pipeline ingests a list of images and a list of possible classes. A class is returned for each of the provided images. For caption generation, only an image file is required.

If you don't have images ready, pull down the sample images using the following commands:

```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

```bash
wget -O buddy.jpeg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/tests/deepsparse/pipelines/sample_images/buddy.jpeg
```

```bash
wget -O thailand.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolact/sample_images/thailand.jpg
```

<p float="left">
  <img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg" width="300" />
  <img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/tests/deepsparse/pipelines/sample_images/buddy.jpeg" width="300" /> 
  <img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolact/sample_images/thailand.jpg" width="300" />
</p>

This will pull down 3 images, a happy dog, St.Peter's basilica, and two elephants.

#### Zero-shot Prediction

Let's run an example to clasify the images. We'll provide the images in a list with their file names as well as a list of possible classes. We'll also provide paths to the exported ONNX models under the `zeroshot_research` root folder.

```python
import numpy as np

from deepsparse import BasePipeline
from deepsparse.clip import (
    CLIPTextInput,
    CLIPVisualInput,
    CLIPZeroShotInput
)

possible_classes = ["ice cream", "an elephant", "a dog", "a building", "a church"]
images = ["basilica.jpg", "buddy.jpeg", "thailand.jpg"]

model_path_text = "zeroshot_research/clip_text.onnx"
model_path_visual = "zeroshot_research/clip_visual.onnx"

kwargs = {
    "visual_model_path": model_path_visual,
    "text_model_path": model_path_text,
}
pipeline = BasePipeline.create(task="clip_zeroshot", **kwargs)

pipeline_input = CLIPZeroShotInput(
    image=CLIPVisualInput(images=images),
    text=CLIPTextInput(text=possible_classes),
)

output = pipeline(pipeline_input).text_scores
for i in range(len(output)):
    prediction = possible_classes[np.argmax(output[i])]
    print(f"Image {images[i]} is a picture of {prediction}")
```

Running the code above, we get the following outuput:

```
DeepSparse, Copyright 2021-present / Neuralmagic, Inc. version: 1.6.0.20230727 COMMUNITY | (3cb4a3e5) (optimized) (system=avx2, binary=avx2)

Image basilica.jpg is a picture of a church
Image buddy.jpeg is a picture of a dog
Image thailand.jpg is a picture of an elephant
```

#### Caption Generation
Let's try a caption generation example. We'll leverage the `thailand.jpg` file that was pulled down earlier. We'll also provide the 3 exported CoCa ONNX models under the `caption_models` folder.

```python
from deepsparse import BasePipeline
from deepsparse.clip import CLIPCaptionInput, CLIPVisualInput

root = "caption_models"
model_path_visual = f"{root}/clip_visual.onnx"
model_path_text = f"{root}/clip_text.onnx"
model_path_decoder = f"{root}/clip_text_decoder.onnx"
engine_args = {"num_cores": 8}

kwargs = {
    "visual_model_path": model_path_visual,
    "text_model_path": model_path_text,
    "decoder_model_path": model_path_decoder,
    "pipeline_engine_args": engine_args
}
pipeline = BasePipeline.create(task="clip_caption", **kwargs)

pipeline_input = CLIPCaptionInput(image=CLIPVisualInput(images="thailand.jpg"))
output = pipeline(pipeline_input).caption
print(output[0])
```
Running the code above, we get the following caption:

```
DeepSparse, Copyright 2021-present / Neuralmagic, Inc. version: 1.6.0.20230727 COMMUNITY | (3cb4a3e5) (optimized) (system=avx2, binary=avx2)

an adult elephant and a baby elephant .
```