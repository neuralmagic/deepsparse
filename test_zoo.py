from sparsezoo import Model

zoo_stub = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned85_quant-none-vnni"
zoo_model = Model(zoo_stub)
data_originals_path = None
if zoo_model.sample_originals is not None:
    if not zoo_model.sample_originals.files:
        zoo_model.sample_originals.unzip()
    data_originals_path = zoo_model.sample_originals.path