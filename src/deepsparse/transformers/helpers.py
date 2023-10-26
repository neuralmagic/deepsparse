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
from deepsparse.utils.onnx import _MODEL_DIR_ONNX_NAME, truncate_onnx_model
from sparsezoo import Model
from sparsezoo.utils import save_onnx


__all__ = [
    "get_deployment_path",
    "overwrite_transformer_onnx_model_inputs",
    "fix_numpy_types",
    "get_transformer_layer_init_names",
    "truncate_transformer_onnx_model",
]

_LOGGER = get_main_logger()


def get_deployment_path(model_path: str) -> Tuple[str, str]:
    """
    Returns the path to the deployment directory
    for the given model path and the path to the mandatory
    ONNX model that should reside in the deployment directory.
    The deployment directory contains all the necessary files
    for running the transformers model in the deepsparse pipeline

    :param model_path: path to model directory, sparsezoo stub, or ONNX file
    :return: path to the deployment directory and path to the ONNX file inside
        the deployment directory
    """
    if os.path.isfile(model_path):
        # return the parent directory of the ONNX file
        return os.path.dirname(model_path), model_path

    if os.path.isdir(model_path):
        model_files = os.listdir(model_path)

        if _MODEL_DIR_ONNX_NAME not in model_files:
            raise ValueError(
                f"{_MODEL_DIR_ONNX_NAME} not found in transformers model directory "
                f"{model_path}. Be sure that an export of the model is written to "
                f"{os.path.join(model_path, _MODEL_DIR_ONNX_NAME)}"
            )
        return model_path, os.path.join(model_path, _MODEL_DIR_ONNX_NAME)

    elif model_path.startswith("zoo:"):
        zoo_model = Model(model_path)
        deployment_path = zoo_model.deployment_directory_path
        return deployment_path, os.path.join(deployment_path, _MODEL_DIR_ONNX_NAME)
    elif model_path.startswith("hf:"):
        from huggingface_hub import snapshot_download

        deployment_path = snapshot_download(repo_id=model_path.replace("hf:", "", 1))
        onnx_path = os.path.join(deployment_path, _MODEL_DIR_ONNX_NAME)
        if not os.path.isfile(onnx_path):
            raise ValueError(
                f"{_MODEL_DIR_ONNX_NAME} not found in transformers model directory "
                f"{deployment_path}. Be sure that an export of the model is written to "
                f"{onnx_path}"
            )
        return deployment_path, onnx_path
    else:
        raise ValueError(
            f"model_path {model_path} is not a valid file, directory, or zoo stub"
        )


def overwrite_transformer_onnx_model_inputs(
    path: str,
    batch_size: int = 1,
    max_length: int = 128,
    inplace: bool = True,
) -> Tuple[Optional[str], List[str], Optional[NamedTemporaryFile]]:
    """
    Overrides an ONNX model's inputs to have the given batch size and sequence lengths.
    Assumes that these are the first and second shape indices of the given model inputs
    respectively

    :param path: path to the ONNX model to override
    :param batch_size: batch size to set
    :param max_length: max sequence length to set
    :param inplace: if True, the model will be modified in place (its inputs will
        be overwritten). Else, a copy of that model, with overwritten inputs,
        will be saved to a temporary file
    :return: tuple of (path to the overwritten model, list of input names that were
        overwritten, and a temporary file containing the overwritten model if
        `inplace=False`, else None)
    """
    # overwrite input shapes
    # if > 2Gb model is to be modified in-place, operate
    # exclusively on the model graph
    model = onnx.load(path, load_external_data=not inplace)
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
    if inplace:
        _LOGGER.debug(
            f"Overwriting in-place the input shapes of the transformer model at {path}"
        )
        save_onnx(model, path)
        return path, input_names, None
    else:
        tmp_file = NamedTemporaryFile()
        _LOGGER.info(
            f"Saving a copy of the transformer model: {path} "
            f"with overwritten input shapes to {tmp_file.name}"
        )
        save_onnx(model, tmp_file.name)
        return tmp_file.name, input_names, tmp_file


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
