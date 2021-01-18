import os

import numpy
import onnx

from deepsparse.utils.log import log_init


__all__ = [
    "get_input_names",
    "get_output_names",
    "generate_random_inputs",
    "override_batch_size",
]

log = log_init(os.path.basename(__file__))

onnx_tensor_type_map = {
    1: numpy.float32,
    2: numpy.uint8,
    3: numpy.int8,
    4: numpy.uint16,
    5: numpy.int16,
    6: numpy.int32,
    7: numpy.int64,
    10: numpy.float16,
    11: numpy.float64,
    12: numpy.uint32,
    13: numpy.uint64,
}


def translate_onnx_type_to_numpy(tensor_type):
    if tensor_type not in onnx_tensor_type_map:
        raise Exception("Unknown ONNX tensor type = {}".format(tensor_type))
    return onnx_tensor_type_map[tensor_type]


def get_input_names(onnx_filepath: str):
    model = onnx.load(onnx_filepath)
    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    input_names = [
        input.name for input in all_inputs if input.name not in initializer_input_names
    ]
    return input_names


def get_output_names(onnx_filepath: str):
    model = onnx.load(onnx_filepath)
    ret = []
    for output_obj in model.graph.output:
        ret.append(output_obj.name)
    return ret


def generate_random_inputs(onnx_filepath: str, batch_size: int = None):
    model = onnx.load(onnx_filepath)

    all_inputs = model.graph.input
    initializer_input_names = [node.name for node in model.graph.initializer]
    external_inputs = [
        input for input in all_inputs if input.name not in initializer_input_names
    ]

    log.info("Generating {} random inputs".format(len(external_inputs)))

    input_data_list = []
    for i, external_input in enumerate(external_inputs):
        in_shape = [int(d.dim_value) for d in external_input.type.tensor_type.shape.dim]

        if batch_size is not None:
            in_shape[0] = batch_size

        log.info("-- random input #{} of shape = {}".format(i, in_shape))
        input_data_list.append(
            numpy.random.rand(*in_shape).astype(
                translate_onnx_type_to_numpy(external_input.type.tensor_type.elem_type)
            )
        )
    return input_data_list


def override_batch_size(onnx_filepath: str, batch_size: int) -> str:
    return onnx_filepath
