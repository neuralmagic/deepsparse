# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx

import pytest
from deepsparse.transformers.helpers import (
    get_transformer_layer_init_names,
    truncate_transformer_onnx_model,
)
from sparsezoo import Model


@pytest.fixture(scope="session")
def model_stubs():
    return {
        "bert": (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/"
            "pruned95_obs_quant-none"
        ),
        "distilbert": (
            "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/"
            "sst2/base-none"
        ),
        "obert": (
            "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/"
            "base-none"
        ),
    }


@pytest.fixture(scope="session")
def get_model_onnx_path(model_stubs):
    model_onnx_paths = {}
    for model_name, model_stub in model_stubs.items():
        model = Model(model_stub)
        onnx_path = model.onnx_model.path
        model_onnx_paths[model_name] = onnx_path

    def _get_model_onnx_path(model_name):
        return model_onnx_paths[model_name]

    return _get_model_onnx_path


@pytest.fixture()
def get_onnx_final_node():
    def _get_onnx_final_node(model_onnx, output_name):
        return next(
            (node for node in model_onnx.graph.node if node.output[0] == output_name),
            None,
        )

    return _get_onnx_final_node


@pytest.mark.parametrize(
    "model_name,expected_init_names",
    [
        (
            "bert",
            [
                "bert.encoder.layer.0.output.LayerNorm.bias",
                "bert.encoder.layer.1.output.LayerNorm.bias",
                "bert.encoder.layer.2.output.LayerNorm.bias",
                "bert.encoder.layer.3.output.LayerNorm.bias",
                "bert.encoder.layer.4.output.LayerNorm.bias",
                "bert.encoder.layer.5.output.LayerNorm.bias",
                "bert.encoder.layer.6.output.LayerNorm.bias",
                "bert.encoder.layer.7.output.LayerNorm.bias",
                "bert.encoder.layer.8.output.LayerNorm.bias",
                "bert.encoder.layer.9.output.LayerNorm.bias",
                "bert.encoder.layer.10.output.LayerNorm.bias",
                "bert.encoder.layer.11.output.LayerNorm.bias",
            ],
        ),
        (
            "distilbert",
            [
                "distilbert.transformer.layer.0.output_layer_norm.bias",
                "distilbert.transformer.layer.1.output_layer_norm.bias",
                "distilbert.transformer.layer.2.output_layer_norm.bias",
                "distilbert.transformer.layer.3.output_layer_norm.bias",
                "distilbert.transformer.layer.4.output_layer_norm.bias",
                "distilbert.transformer.layer.5.output_layer_norm.bias",
            ],
        ),
        (
            "obert",
            [
                "bert.encoder.layer.0.output.LayerNorm.bias",
                "bert.encoder.layer.1.output.LayerNorm.bias",
                "bert.encoder.layer.2.output.LayerNorm.bias",
                "bert.encoder.layer.3.output.LayerNorm.bias",
                "bert.encoder.layer.4.output.LayerNorm.bias",
                "bert.encoder.layer.5.output.LayerNorm.bias",
                "bert.encoder.layer.6.output.LayerNorm.bias",
                "bert.encoder.layer.7.output.LayerNorm.bias",
                "bert.encoder.layer.8.output.LayerNorm.bias",
                "bert.encoder.layer.9.output.LayerNorm.bias",
                "bert.encoder.layer.10.output.LayerNorm.bias",
                "bert.encoder.layer.11.output.LayerNorm.bias",
            ],
        ),
    ],
)
def test_get_transformer_layer_init_names(
    model_name,
    expected_init_names,
    get_model_onnx_path,
):
    model_onnx_path = get_model_onnx_path(model_name)
    model = onnx.load(model_onnx_path)

    init_names = get_transformer_layer_init_names(model)
    assert init_names == expected_init_names


@pytest.mark.parametrize(
    "model_name,emb_extraction_layer,expected_final_node_name",
    [
        ("bert", -1, "Add_2544"),
        ("bert", 5, "Add_1296"),
        ("bert", 0, "Add_256"),
        ("bert", "Add_256", "Add_256"),
        ("distilbert", -1, "Add_515"),
        ("distilbert", 2, "Add_269"),
        ("distilbert", 0, "Add_105"),
        ("distilbert", "Add_105", "Add_105"),
        ("obert", -1, "Add_1158"),
        ("obert", 5, "Add_594"),
        ("obert", 0, "Add_124"),
        ("obert", "Add_124", "Add_124"),
    ],
)
def test_truncate_transformer_onnx_model(
    model_name,
    emb_extraction_layer,
    expected_final_node_name,
    get_model_onnx_path,
    get_onnx_final_node,
):
    model_onnx_path = get_model_onnx_path(model_name)
    output_name = "embedding"

    (truncated_onnx_path, output_names, _,) = truncate_transformer_onnx_model(
        model_path=model_onnx_path,
        emb_extraction_layer=emb_extraction_layer,
        hidden_layer_size=None,
        output_name=output_name,
        output_path=None,
    )

    truncated_onnx = onnx.load(truncated_onnx_path)
    assert len(truncated_onnx.graph.output) == 1
    assert truncated_onnx.graph.output[0].name == output_name
    final_node = get_onnx_final_node(truncated_onnx, output_name)
    assert final_node.name == expected_final_node_name
