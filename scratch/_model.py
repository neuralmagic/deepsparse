from sparsezoo import File, Model
path = "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate"

model = Model(path)

print(model)