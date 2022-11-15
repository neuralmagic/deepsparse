import os

import onnx
import pytest
from sparsezoo import Model


@pytest.fixture(scope="session")
def stub():
    yield "zoo:cv/classification/mnistnet/pytorch/sparseml/mnist/base-none"


@pytest.fixture(scope="session")
def onnx_filepath(stub):
    path = Model(source=stub).deployment.path
    yield os.path.join(path, "model.onnx")


@pytest.fixture(scope="session")
def onnx_model(onnx_filepath):
    yield onnx.load(onnx_filepath)


def test_truncate_onnx():
    pass


def test_get_sorted_layer_init_names(onnx_model):
    init_names = test_get_sorted_layer_init_names(onnx_model=onnx_model)
