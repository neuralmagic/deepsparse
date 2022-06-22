from deepsparse import compile_model
import numpy

from deepsparse import Pipeline
cv_pipeline = Pipeline.create(
  task='yolact',
  #model_path='/home/damian/yolact/yolact/yolact.onnx',
  model_path = "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none"
)

inference = cv_pipeline(images=["golfish.jpeg"])