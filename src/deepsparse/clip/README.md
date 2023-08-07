# CLIP Inference Pipelines

DeepSparse allows inference on [CLIP](https://github.com/mlfoundations/open_clip) models.  

The CLIP integration currently supports the following task:
- **Zero-shot Image Classification** - Classifying images given possible classes

## Getting Started

Before you start your adventure with the DeepSparse Engine, make sure that your machine is compatible with our [hardware requirements](https://docs.neuralmagic.com/deepsparse/source/hardware.html).

### Installation
```pip install deepsparse[clip]```

### Model Format
By default, to deploy CLIP models using the DeepSparse Engine, it is required to supply the model in the ONNX format. This grants the engine the flexibility to serve any model in a framework-agnostic environment. To see examples of pulling CLIP models and exporting them to ONNX, please see the [sparseml documentation](https://github.com/neuralmagic/sparseml/tree/main/integrations/clip). For the Zero-shot image classification workflow, two ONNX models are required, a visual model for CLIP's visual branch, and a text model for CLIP's text branch. Both of these model should be produced through the sparseml integration linked above.

### Deployment examples:
The following example uses pipelines to run the CLIP models for inference. As input, the pipeline ingests a list of images and a list of possible classes. A class is returned for each of the provided images.

If you don't have images ready, pull down the sample images using the following commands:

```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg

wget -O buddy.jpeg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/tests/deepsparse/pipelines/sample_images/buddy.jpeg
```

This will pull down two images, one with a happy dog and one with St.Peter's basilica. 

#### Zero-shot Prediction

Let's run an example to clasify the images. We'll provide the images in a list with their file names as well as a list of possible classes. We'll also provide paths to the exported ONNX models.

```python
import numpy as np

from deepsparse import BasePipeline
from deepsparse.clip import (
    CLIPTextInput,
    CLIPVisualInput,
    CLIPZeroShotInput
)

possible_classes = ["ice cream", "an elephant", "a dog", "a building", "a church"]
images = ["basilica.jpg", "buddy.jpeg"]

model_path_text = "zeroshot_research/text/model.onnx"
model_path_visual = "zeroshot_research/visual/model.onnx"

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
```