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

"""
Helper functions for working with ONNX exports of transformer models and deepsparse
"""


import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple, Union

import numpy
import onnx
from onnx import ModelProto

from deepsparse.log import get_main_logger
from deepsparse.utils.onnx import truncate_onnx_model
from sparsezoo import Model


__all__ = [
    "get_onnx_path_and_configs",
    "overwrite_transformer_onnx_model_inputs",
    "fix_numpy_types",
    "get_transformer_layer_init_names",
    "truncate_transformer_onnx_model",
]

_LOGGER = get_main_logger()

_MODEL_DIR_ONNX_NAME = "model.onnx"
_MODEL_DIR_CONFIG_NAME = "config.json"
_MODEL_DIR_TOKENIZER_NAME = "tokenizer.json"
_MODEL_DIR_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"


def get_onnx_path_and_configs(
    model_path: str,
    require_configs: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    :param model_path: path to onnx file, transformers sparsezoo stub,
        or directory containing `model.onnx`, `config.json`, and/or
        `tokenizer.json` files. If no `model.onnx` file is found in
        a model directory, an exception will be raised
    :param require_configs: if True, model_path must be a directory containing
        `model.onnx`, `config.json`, and `tokenizer.json` files. Will raise
        an exception otherwise
    :return: tuple of ONNX file path, parent directory of config file
        if it exists, and parent directory of tokenizer config file if it
        exists. (Parent directories returned instead of absolute path
        for compatibility with transformers .from_pretrained() method)
    """
    if os.path.isfile(model_path) and not require_configs:
        return model_path, None, None

    config_path = None
    tokenizer_path = None

    if os.path.isdir(model_path):
        model_files = os.listdir(model_path)

        if _MODEL_DIR_ONNX_NAME not in model_files:
            raise ValueError(
                f"{_MODEL_DIR_ONNX_NAME} not found in transformers model directory "
                f"{model_path}. Be sure that an export of the model is written to "
                f"{os.path.join(model_path, _MODEL_DIR_ONNX_NAME)}"
            )
        onnx_path = os.path.join(model_path, _MODEL_DIR_ONNX_NAME)

        # attempt to read config and tokenizer from sparsezoo-like framework directory
        framework_dir = None
        if "framework" in model_files:
            framework_dir = os.path.join(model_path, "framework")
        if "pytorch" in model_files:
            framework_dir = os.path.join(model_path, "pytorch")
        if framework_dir and os.path.isdir(framework_dir):
            framework_files = os.listdir(framework_dir)
            if _MODEL_DIR_CONFIG_NAME in framework_files:
                config_path = framework_dir
            if _MODEL_DIR_TOKENIZER_NAME in framework_files:
                tokenizer_path = framework_dir

        # prefer config and tokenizer files in same directory as model.onnx
        if _MODEL_DIR_CONFIG_NAME in model_files:
            config_path = model_path
        if _MODEL_DIR_TOKENIZER_NAME in model_files:
            tokenizer_path = model_path

    elif model_path.startswith("zoo:"):
        zoo_model = Model(model_path)
        onnx_path = zoo_model.onnx_model.path
        config_path = _get_file_parent(
            zoo_model.deployment.default.get_file(_MODEL_DIR_CONFIG_NAME).path
        )
        tokenizer_path = _get_file_parent(
            zoo_model.deployment.default.get_file(_MODEL_DIR_TOKENIZER_NAME).path
        )
        tokenizer_config_path = zoo_model.deployment.default.get_file(
            _MODEL_DIR_TOKENIZER_CONFIG_NAME
        )
        if tokenizer_config_path is not None:
            tokenizer_config_path.path  # trigger download of tokenizer_config
    elif require_configs and (config_path is None or tokenizer_path is None):
        raise RuntimeError(
            f"Unable to find model and tokenizer config for model_path {model_path}. "
            f"model_path must be a directory containing model.onnx, config.json, and "
            f"tokenizer.json files. Found config and tokenizer paths: {config_path}, "
            f"{tokenizer_path}"
        )
    else:
        raise ValueError(
            f"model_path {model_path} is not a valid file, directory, or zoo stub"
        )

    return onnx_path, config_path, tokenizer_path


def overwrite_transformer_onnx_model_inputs(
    path: str,
    batch_size: int = 1,
    max_length: int = 128,
    output_path: Optional[str] = None,
) -> Tuple[Optional[str], List[str], Optional[NamedTemporaryFile]]:
    """
    Overrides an ONNX model's inputs to have the given batch size and sequence lengths.
    Assumes that these are the first and second shape indices of the given model inputs
    respectively

    :param path: path to the ONNX model to override
    :param batch_size: batch size to set
    :param max_length: max sequence length to set
    :param output_path: if provided, the model will be saved to the given path,
        otherwise, the model will be saved to a named temporary file that will
        be deleted after the program exits
    :return: if no output path, a tuple of the saved path to the model, list of
        model input names, and reference to the tempfile object will be returned
        otherwise, only the model input names will be returned
    """
    # overwrite input shapes
    model = onnx.load(path)
    initializer_input_names = set([node.name for node in model.graph.initializer])
    external_inputs = [
        inp for inp in model.graph.input if inp.name not in initializer_input_names
    ]
    input_names = []
    for external_input in external_inputs:
        external_input.type.tensor_type.shape.dim[0].dim_value = batch_size
        external_input.type.tensor_type.shape.dim[1].dim_value = max_length
        input_names.append(external_input.name)

    # Save modified model
    if output_path is None:
        tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
        onnx.save(model, tmp_file.name)

        return tmp_file.name, input_names, tmp_file
    else:
        onnx.save(model, output_path)
        return input_names


def _get_file_parent(file_path: str) -> str:
    return str(Path(file_path).parent.absolute())


def fix_numpy_types(func):
    """
    Decorator to fix numpy types in Dicts, List[Dicts], List[List[Dicts]]
    Because `orjson` does not support serializing individual numpy data types
    yet
    """

    def _wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        def _normalize_fields(_dict):
            if isinstance(_dict, dict):
                for field in _dict:
                    if isinstance(_dict[field], numpy.generic):
                        _dict[field] = _dict[field].item()

        if isinstance(result, dict):
            _normalize_fields(result)
        elif result and isinstance(result, list):
            for element in result:
                if isinstance(element, list):
                    for _result in element:
                        _normalize_fields(_result)
                else:
                    _normalize_fields(element)

        return result

    return _wrapper


def get_transformer_layer_init_names(model: ModelProto) -> List[str]:
    """
    Attempts to find the names of the initializers corresponding to the last nodes
    in a bert or distillbert layer. Throws RuntimeError if cannot find initializer
    names matching the expected formats.

    :param model: model containing bert or distilbert layers
    :return: a list of initializer names belonging to nodes at the end of the
        transformer layer
    """
    bert_layer_pattern = re.compile(
        r"bert\.encoder\.layer\.\d+\.output\.LayerNorm\.bias"
    )
    distillbert_layer_pattern = re.compile(
        r"distilbert.transformer.layer.\d+\.output_layer_norm.bias"
    )

    layer_init_names = [
        initializer.name
        for initializer in model.graph.initializer
        if bert_layer_pattern.match(initializer.name)
        or distillbert_layer_pattern.match(initializer.name)
    ]
    if len(layer_init_names) <= 0:
        raise RuntimeError(
            "Unable to find bert layers within onnx graph using initializer "
            "name matching"
        )

    return sorted(
        layer_init_names,
        key=lambda name: int(re.findall("[0-9]+", name)[0]),
        reverse=False,
    )


def truncate_transformer_onnx_model(
    model_path: str,
    emb_extraction_layer: Union[int, str] = -1,
    hidden_layer_size: Optional[int] = None,
    output_name: str = "embedding",
    output_path: Optional[str] = None,
) -> Tuple[str, List[str], Union[NamedTemporaryFile, None]]:
    """
    Determines where to cut the transformer model using best-guess heuristics
    Saves cut model to output_path or temporary file

    :param model_path: path of onnx file to be cut
    :param emb_extraction_layer: if an int, last bert layer to include. If a
        string, then the name of the last node in the truncated graph.
        default -1 (last layer)
    :param hidden_layer_size: guess for the number of embedded values per token
        in provided model. Used by deepsparse engine to optimize memory allocation
    :param output_name: name of graph output, default "embedding"
    :param output_path: path to write resulting onnx file. If not provided,
        will create a temporary file path that will be destroyed on program end
    :return: if no output path, a tuple of the saved path to the model, list of
        model output names, and reference to the tempfile object will be returned
        otherwise, a tuple containing the given output_path argument, the model
        output names, and None
    """

    # determine where to cut the model
    final_node_name = (
        emb_extraction_layer if isinstance(emb_extraction_layer, str) else None
    )
    if final_node_name is None:
        model = onnx.load(model_path)

        # try to match bert layers by initializer names
        try:
            layer_init_names_sorted = get_transformer_layer_init_names(model)
            final_node_initializer_name = layer_init_names_sorted[emb_extraction_layer]
            final_node_name = [
                node.name
                for node in model.graph.node
                if final_node_initializer_name in node.input
            ][0]
        except Exception as exception:
            raise RuntimeError(f"Failed to truncate transformer: {exception}")

    # create temporary file if necessary
    if output_path is None:
        tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
        output_filepath = tmp_file.name
        tmp_file_or_none = tmp_file
    else:
        output_filepath = output_path
        tmp_file_or_none = None

    # create subgraph
    truncate_onnx_model(
        onnx_filepath=model_path,
        output_filepath=output_filepath,
        final_node_names=[final_node_name],
        graph_output_names=[output_name],
        graph_output_shapes=[[None, hidden_layer_size]],
    )

    return output_filepath, [output_name], tmp_file_or_none
