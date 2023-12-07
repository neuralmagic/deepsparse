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

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple, Union

import numpy
import onnx
from onnx import ModelProto
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from deepsparse.utils import parse_input_shapes
from deepsparse.utils.extractor import Extractor
from sparsezoo.utils import save_onnx, validate_onnx


try:
    from sparsezoo import File, Model

    sparsezoo_import_error = None
except Exception as sparsezoo_err:
    Model = object
    File = object
    sparsezoo_import_error = sparsezoo_err

__all__ = [
    "model_to_path",
    "get_external_inputs",
    "get_external_outputs",
    "get_input_names",
    "get_output_names",
    "generate_random_inputs",
    "override_onnx_batch_size",
    "override_onnx_input_shapes",
    "truncate_onnx_model",
    "truncate_onnx_embedding_model",
    "overwrite_onnx_model_inputs_for_kv_cache_models",
    "default_cached_outputs",
    "infer_sequence_length",
    "has_model_kv_cache",
    "CACHE_INPUT_PREFIX",
    "CACHE_OUTPUT_PREFIX",
    "_MODEL_DIR_ONNX_NAME",
]

_LOGGER = logging.getLogger(__name__)

_MODEL_DIR_ONNX_NAME = "model.onnx"
CACHE_INPUT_PREFIX = "past_key_values"
CACHE_OUTPUT_PREFIX = "present"


@contextlib.contextmanager
def save_onnx_to_temp_files(model: onnx.ModelProto, with_external_data=False) -> str:
    """
    Save model to a temporary file. Works for models with external data.

    :param model: The onnx model to save to temporary directory
    :param with_external_data: Whether to save external data to a separate file
    """

    shaped_model = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False, mode="w")
    _LOGGER.info(f"Saving model to temporary file: {shaped_model.name}")

    if with_external_data:
        external_data = tempfile.NamedTemporaryFile(
            suffix=".data", delete=False, mode="w"
        )
        _LOGGER.info(f"Saving external data to temporary file: {external_data.name}")
        has_external_data = save_onnx(model, shaped_model.name, external_data.name)
    else:
        has_external_data = save_onnx(model, shaped_model.name)
    try:
        yield shaped_model.name
    finally:
        os.unlink(shaped_model.name)
        shaped_model.close()
        if has_external_data:
            os.unlink(external_data)


def translate_onnx_type_to_numpy(tensor_type: int):
    """
    Translates ONNX types to numpy types
    :param tensor_type: Integer representing a type in ONNX spec
    :return: Corresponding numpy type
    """
    if tensor_type not in TENSOR_TYPE_TO_NP_TYPE:
        raise Exception("Unknown ONNX tensor type = {}".format(tensor_type))
    return TENSOR_TYPE_TO_NP_TYPE[tensor_type]


def model_to_path(model: Union[str, Model, File]) -> str:
    """
    Deals with the various forms a model can take. Either an ONNX file,
    a directory containing model.onnx, a SparseZoo model stub prefixed by
    'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File object that
    defines the neural network. Noting the model will be downloaded automatically
    if a SparseZoo stub is passed

    :param model: Either a local str path or SparseZoo stub to the model. Can
        also be a sparsezoo.Model or sparsezoo.File object
    :returns: The absolute local str path to the model
    """
    if not model:
        raise ValueError("model must be a path, sparsezoo.Model, or sparsezoo.File")

    if isinstance(model, str) and model.startswith("zoo:"):
        # load SparseZoo Model from stub
        if sparsezoo_import_error is not None:
            raise sparsezoo_import_error
        model = Model(model)

    if Model is not object and isinstance(model, Model):
        # trigger download and unzipping of deployment directory if not cached
        model.deployment.path

        # default to the main onnx file for the model
        model = model.deployment.get_file(_MODEL_DIR_ONNX_NAME).path

    elif File is not object and isinstance(model, File):
        # get the downloaded_path -- will auto download if not on local system
        model = model.path

    if isinstance(model, str) and model.startswith("hf:"):
        # load Hugging Face model from stub
        from huggingface_hub import snapshot_download

        deployment_path = snapshot_download(repo_id=model.replace("hf:", "", 1))
        onnx_path = os.path.join(deployment_path, _MODEL_DIR_ONNX_NAME)
        if not os.path.isfile(onnx_path):
            raise ValueError(
                f"Could not find the ONNX model file '{_MODEL_DIR_ONNX_NAME}' in the "
                f"Hugging Face Hub repository located at {deployment_path}. Please "
                f"ensure the model has been correctly exported to ONNX format and "
                f"exists in the repository."
            )
        return onnx_path

    if not isinstance(model, str):
        raise ValueError("unsupported type for model: {}".format(type(model)))

    if not os.path.exists(model):
        raise ValueError("model path must exist: given {}".format(model))

    model_path = Path(model)
    if model_path.is_dir():
        return str(model_path / _MODEL_DIR_ONNX_NAME)

    return model


def get_external_inputs(onnx_model: Union[str, ModelProto]) -> List:
    """
    Gather external inputs of ONNX model
    :param onnx_model: File path to ONNX model or ONNX model object
    :return: List of input objects
    """
    model = (
        onnx_model
        if isinstance(onnx_model, ModelProto)
        else onnx.load(onnx_model, load_external_data=False)
    )
    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    external_inputs = [
        input for input in all_inputs if input.name not in initializer_input_names
    ]
    return external_inputs


def get_external_outputs(onnx_model: Union[str, ModelProto]) -> List:
    """
    Gather external outputs of ONNX model
    :param onnx_model: File path to ONNX model or ONNX model object
    :return: List of output objects
    """
    model = (
        onnx_model
        if isinstance(onnx_model, ModelProto)
        else onnx.load(onnx_model, load_external_data=False)
    )
    return [output for output in model.graph.output]


def get_input_names(onnx_filepath: Union[str, ModelProto]) -> List[str]:
    """
    Gather names of all external inputs of ONNX model
    :param onnx_filepath: File path to ONNX model or ONNX model object
    :return: List of string names
    """
    return [input_.name for input_ in get_external_inputs(onnx_filepath)]


def get_output_names(onnx_filepath: str) -> List[str]:
    """
    Gather names of all external outputs of ONNX model
    :param onnx_filepath: File path to ONNX model or ONNX model object
    :return: List of string names
    """
    return [output.name for output in get_external_outputs(onnx_filepath)]


def generate_random_inputs(
    onnx_filepath: Union[str, ModelProto], batch_size: int = None
) -> List[numpy.array]:
    """
    Generate random data that matches the type and shape of ONNX model,
    with a batch size override
    :param onnx_filepath: File path to ONNX model or ONNX model object
    :param batch_size: If provided, override for the batch size dimension
    :return: List of random tensors
    """
    input_data_list = []
    for i, external_input in enumerate(get_external_inputs(onnx_filepath)):
        input_tensor_type = external_input.type.tensor_type
        elem_type = translate_onnx_type_to_numpy(input_tensor_type.elem_type)
        in_shape = [max(int(d.dim_value), 1) for d in input_tensor_type.shape.dim]

        if batch_size is not None:
            in_shape[0] = batch_size

        input_string = "input '{}', type = {}, shape = {}".format(
            external_input.name, numpy.dtype(elem_type).name, in_shape
        )

        assert not any(dim < 1 for dim in in_shape), (
            f"Dynamic shape found in {input_string}. "
            "All shapes must be non-zero in order to generate random data"
        )

        _LOGGER.info(f"Generating {input_string}")
        input_data_list.append(numpy.random.rand(*in_shape).astype(elem_type))
    return input_data_list


@contextlib.contextmanager
def override_onnx_batch_size(
    onnx_filepath: str,
    batch_size: int,
    inplace: bool = True,
) -> str:
    """
    Rewrite batch sizes of ONNX model, saving the modified model and returning its path

    :param onnx_filepath: File path to ONNX model. If the graph is to be
        modified in-place, only the model graph will be loaded and modified.
        Otherwise, the entire model will be loaded and modified, so that
        external data are saved along the model graph.
    :param batch_size: Override for the batch size dimension
    :param inplace: If True, overwrite the original model file.
        Else, save the modified model to a temporary file.
    :return: File path to modified ONNX model.
        If inplace is True,
        the modified model will be saved to the same path as the original
        model. Else the modified model will be saved to a
        temporary file.
    """

    if batch_size is None:
        return onnx_filepath

    model = onnx.load(onnx_filepath, load_external_data=not inplace)
    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    external_inputs = [
        input for input in all_inputs if input.name not in initializer_input_names
    ]
    for external_input in external_inputs:
        external_input.type.tensor_type.shape.dim[0].dim_value = batch_size

    if inplace:
        _LOGGER.debug(
            f"Overwriting in-place the batch size of the model at {onnx_filepath}"
        )
        save_onnx(model, onnx_filepath)
        yield onnx_filepath
    else:
        with save_onnx_to_temp_files(
            model, with_external_data=not inplace
        ) as temp_file:
            yield temp_file


@contextlib.contextmanager
def override_onnx_input_shapes(
    onnx_filepath: str,
    input_shapes: Union[None, str, List[int], List[List[int]]],
    inplace: bool = True,
) -> str:
    """
    Rewrite input shapes of ONNX model, saving the modified model and returning its path

    :param onnx_filepath: File path to ONNX model. If the graph is to be
        modified in-place, only the model graph will be loaded and modified.
        Otherwise, the entire model will be loaded and modified, so that
        external data are saved along the model graph.
    :param input_shapes: Override for model's input shapes
    :param inplace: If True, overwrite the original model file
    :return: File path to modified ONNX model.
        If inplace is True,
        the modified model will be saved to the same path as the original
        model. Else the modified model will be saved to a
        temporary file.
    """

    if input_shapes is None:
        return onnx_filepath

    model = onnx.load(onnx_filepath, load_external_data=not inplace)
    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    external_inputs = [
        input for input in all_inputs if input.name not in initializer_input_names
    ]

    if isinstance(input_shapes, str):
        input_shapes = parse_input_shapes(input_shapes)

    # Input shapes should be a list of lists, even if there is only one input
    if not all(isinstance(inp, list) for inp in input_shapes):
        input_shapes = [input_shapes]

    # If there is a single input shape given and multiple inputs,
    # duplicate for all inputs to apply the same shape
    if len(input_shapes) == 1 and len(external_inputs) > 1:
        input_shapes.extend([input_shapes[0] for _ in range(1, len(external_inputs))])

    # Make sure that input shapes can map to the ONNX model
    assert len(external_inputs) == len(
        input_shapes
    ), "Mismatch of number of model inputs ({}) and override shapes ({})".format(
        len(external_inputs), len(input_shapes)
    )

    # Overwrite the input shapes of the model
    for input_idx, external_input in enumerate(external_inputs):
        assert len(external_input.type.tensor_type.shape.dim) == len(
            input_shapes[input_idx]
        ), "Input '{}' shape doesn't match shape override: {} vs {}".format(
            external_input.name,
            external_input.type.tensor_type.shape.dim,
            input_shapes[input_idx],
        )
        for dim_idx, dim in enumerate(external_input.type.tensor_type.shape.dim):
            dim.dim_value = input_shapes[input_idx][dim_idx]

    if inplace:
        _LOGGER.debug(
            f"Overwriting in-place the input shapes of the model at {onnx_filepath}"
        )
        onnx.save(model, onnx_filepath)
        yield onnx_filepath
    else:
        _LOGGER.info(
            f"Saving the input shapes of the model at {onnx_filepath} "
            f"to a temporary file"
        )
        with save_onnx_to_temp_files(
            model, with_external_data=not inplace
        ) as temp_file:
            yield temp_file


def truncate_onnx_model(
    onnx_filepath: str,
    output_filepath: str,
    final_node_names: List[str],
    graph_output_names: List[str],
    graph_output_shapes: Optional[List[List[int]]] = None,
) -> None:
    """
    :param onnx_filepath: file path to onnx model
    :param output_filepath: file path to save new onnx model
    :param final_node_names: list of node names whose outputs will become the
        outputs of the graph
    :param graph_output_names: list of names to call the graph outputs. Names
        correspond with the outputs specified in final_node_names
    :param graph_output_shapes: list of shapes for each output. If not provided,
        defaults to [None] for each output and leads to slight performance loss
    :return: None
    """
    if graph_output_shapes is None:
        graph_output_shapes = [None] * len(final_node_names)

    if len(final_node_names) != len(graph_output_names) != len(graph_output_shapes):
        raise ValueError(
            f"length of final_node_names {len(final_node_names)}, "
            f"graph_output_names {len(graph_output_names)}, and "
            f"graph_output_shapes {len(graph_output_shapes)} must all match"
        )

    if len(set(final_node_names)) != len(final_node_names):
        raise ValueError("final_node_names must not contain duplicate names")

    if len(set(graph_output_names)) != len(graph_output_names):
        raise ValueError("graph_output_names must not contain duplicate names")

    model = onnx.load(onnx_filepath)
    final_nodes = [node for node in model.graph.node if node.name in final_node_names]

    if len(final_nodes) != len(final_node_names):
        raise ValueError("Could not find final node names in model graph")

    for final_node, graph_output_name, graph_output_shape in zip(
        final_nodes, graph_output_names, graph_output_shapes
    ):
        # write each node's output to new output
        [final_node.output.pop() for _ in final_node.output]
        final_node.output.append(graph_output_name)

        # write graph output. TODO: use ort to find real shapes and types
        output_value_info = onnx.helper.make_tensor_value_info(
            graph_output_name, onnx.TensorProto.UNDEFINED, graph_output_shape
        )
        model.graph.output.append(output_value_info)

    # collect graph inputs
    graph_input_names = [input.name for input in model.graph.input]

    # use extractor to create and write subgraph
    original_num_nodes = len(model.graph.node)
    extractor = Extractor(model)
    extracted_model = extractor.extract_model(
        input_names=graph_input_names, output_names=graph_output_names
    )
    extracted_num_nodes = len(extracted_model.graph.node)
    _LOGGER.info(
        f"Truncating model graph to {final_node_names}. "
        f"Removed {original_num_nodes - extracted_num_nodes} nodes, "
        f"{extracted_num_nodes} remaining"
    )

    for output in extracted_model.graph.output:
        if len(output.type.tensor_type.shape.dim) == 0:
            # ONNX checker treats None shapes and empty shapes
            # differently, clear None shape to pass checker
            output.type.tensor_type.shape.Clear()

    # save and check model
    _LOGGER.debug(f"Saving truncated model to {output_filepath}")
    save_onnx(extracted_model, output_filepath, "external_data")
    validate_onnx(output_filepath)


def truncate_onnx_embedding_model(
    model_path: str,
    emb_extraction_layer: Union[int, str, None] = None,
    output_filepath: Optional[str] = None,
) -> Tuple[str, Optional[NamedTemporaryFile]]:
    """
     :param model_path: path of onnx file to be cut
    :param emb_extraction_layer: if an int, last layer to include. If a
        string, then the name of the last node in the truncated graph.
        default is None.
    :param output_filepath: path to write resulting onnx file. If not provided,
        will create a temporary file path that will be destroyed on program end
    :return: if no output path, a tuple of the saved path to the model, list of
        model output names, and reference to the tempfile object will be returned
        otherwise, a tuple containing the given output_path argument, the model
        output names, and None
    """

    tmp_file = None
    if output_filepath is None:
        tmp_file = NamedTemporaryFile()
        output_filepath = tmp_file.name

    # determine where to cut the model
    model = onnx.load(model_path)
    if isinstance(emb_extraction_layer, str):
        final_node = None
        for graph_node in model.graph.node:
            if graph_node.name == emb_extraction_layer:
                final_node = graph_node

        if final_node is None:
            raise RuntimeError(
                f"Unable to find node {emb_extraction_layer} for extraction in graph"
            )

        final_node_name = final_node.name
        graph_output_name = final_node.output[0]
    else:
        final_node_name = model.graph.node[emb_extraction_layer].name
        graph_output_name = model.graph.node[emb_extraction_layer].output[0]

        if final_node_name is None:
            raise ValueError(
                f"Node at index {emb_extraction_layer} does not have a name set"
            )

    truncate_onnx_model(
        onnx_filepath=model_path,
        output_filepath=output_filepath,
        final_node_names=[final_node_name],
        graph_output_names=[graph_output_name],
        graph_output_shapes=None,
    )

    return output_filepath, tmp_file


def overwrite_onnx_model_inputs_for_kv_cache_models(
    onnx_file_path: str,
    sequence_length: int,
    input_ids_length: int,
    batch_size: int = 1,
) -> Tuple[str, List[int], Optional[int]]:
    """
    Enforces the appropriate input shapes for the onnx model, as well as
    checks whether kv cache is enabled or not.

    :param onnx_file_path: The path to the onnx model file that will be
        overwritten with the new input shapes
    :param batch_size: The batch size to use for the input
    :param sequence_length: The sequence length to use for the input
    :param input_ids_length: The length of input_ids
    :return: A tuple that contains:
        -   the path to the onnx model file that has been overwritten
            with the new input shapes
        -   boolean list, where elements are set to True if the
            corresponding model output should be cached or False
            if not.
        -   the data type of the kv cache. If the model does not
            use kv cache, then the data type is None
    """
    model = onnx.load(onnx_file_path, load_external_data=False)
    initializer_input_names = set(node.name for node in model.graph.initializer)
    external_inputs = [
        inp for inp in model.graph.input if inp.name not in initializer_input_names
    ]
    for external_input in external_inputs:
        # overwrite the batch size for all the inputs
        external_input.type.tensor_type.shape.dim[0].dim_value = batch_size

        if external_input.name in ["input_ids", "positions"]:
            external_input.type.tensor_type.shape.dim[1].dim_value = input_ids_length
        elif external_input.name == "attention_mask":
            external_input.type.tensor_type.shape.dim[1].dim_value = sequence_length
        elif external_input.name.startswith("past_key_values"):
            external_input.type.tensor_type.shape.dim[2].dim_value = (
                sequence_length - input_ids_length
            )
        elif external_input.name.startswith("causal_mask"):
            external_input.type.tensor_type.shape.dim[2].dim_value = input_ids_length
            external_input.type.tensor_type.shape.dim[3].dim_value = sequence_length
        else:
            raise ValueError(f"Unexpected external input name: {external_input.name}")

    _LOGGER.debug(
        "Overwriting in-place the input shapes "
        f"of the transformer model at {onnx_file_path}"
    )
    save_onnx(model, onnx_file_path)

    output_indices_to_be_cached = [
        1 if inp.name.startswith("present") else 0 for inp in model.graph.output
    ]

    kv_cache_data_type = None
    if any(output_indices_to_be_cached):
        kv_cache_elem_type = next(
            inp for inp in model.graph.input if inp.name.startswith("past_key_values")
        ).type.tensor_type.elem_type
        kv_cache_data_type = translate_onnx_type_to_numpy(kv_cache_elem_type)

    return onnx_file_path, output_indices_to_be_cached, kv_cache_data_type


def default_cached_outputs(model_path: str) -> List[bool]:
    """
    Get a list of bools that indicate which outputs should be cached.
    The elements that are set to True correspond to cached outputs,
    the rest are set to False.

    :param model_path: Path to the model.
    :return A list of bools that indicate which outputs should be cached.
    """

    output_names = get_output_names(model_path)
    assert len(output_names) > 0

    return [name.startswith(CACHE_OUTPUT_PREFIX) for name in output_names]


def has_model_kv_cache(model: Union[str, ModelProto]) -> bool:
    """
    Check whether a model has a KV cache support.

    :param model_path: Path to a model or a model proto.
    :return True if the model has a KV cache support, False otherwise.
    """
    return bool(any(default_cached_outputs(model)))


def infer_sequence_length(model: Union[str, ModelProto]) -> int:
    """
    :param model: model
    :return: inferred sequence length of the model
    """
    if not isinstance(model, ModelProto):
        model = onnx.load(model, load_external_data=False)

    # try to find attention mask dim, default to 0
    target_input_idx = 0
    for idx, inp in enumerate(model.graph.input):
        if inp.name == "attention_mask":
            target_input_idx = idx
    try:
        # return shape of second dim if possible
        target_input = model.graph.input[target_input_idx]
        return target_input.type.tensor_type.shape.dim[1].dim_value
    except Exception:
        return 0  # unable to infer seq len
