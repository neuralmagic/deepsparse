# run_inference.py

from deepsparse import Pipeline
from time import sleep

# SparseZoo model stub or path to ONNX file
model_path = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none"

# logger object referencing the local logging config file
logger = ManagerLogger(config="config.yaml")

# pipeline instantiated with the config file
img_classification_pipeline = Pipeline.create(
    task="image-classification",
    model_path=model_path,
    logger=logger
)

# runs for 2 minutes 
for _ in range(120):
    preds = img_classification_pipeline(["piglet.jpg"])
    print(preds)
    sleep(1)
