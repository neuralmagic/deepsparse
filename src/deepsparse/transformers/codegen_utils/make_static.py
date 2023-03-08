import argparse
import onnx
import os
# Tool to set input dimensions of an ONNX model to static numbers
# This is needed since frameworks like TensorFlow can export models with dynamic input sizes
#
# Example usage: You want to set the input of a model to dimensions [1,227,227,3]
# python tools/set-onnx-input-dims.py ~/Downloads/model.onnx -i 1 227 227 3
parser = argparse.ArgumentParser()
parser.add_argument(
    "onnx_filename", help="The full filepath of the onnx model file",
)
args = parser.parse_args()
onnx_filename = args.onnx_filename
tmp_net = onnx.load(onnx_filename)
for i, graph_input in enumerate(tmp_net.graph.input):
    tensor_type = graph_input.type.tensor_type
    print('----')
    print(graph_input.name)

    if graph_input.name == "input_ids":
        input_shapes = [10, 1]
    elif graph_input.name.startswith("past"):
        input_shapes = [10, 16, 383, 64]
    elif graph_input.name.startswith("attention"):
        input_shapes = [10, 384]
    else:
        raise ValueError("")

    in_shape = [int(dim) for dim in input_shapes]
    for dim_id, dim in enumerate(in_shape):
        print(
            "Setting dim #{} to {} (original value {})".format(
                dim_id, dim, graph_input.type.tensor_type.shape.dim[dim_id].dim_value
            )
        )
        graph_input.type.tensor_type.shape.dim[dim_id].dim_value = dim
name, ext = os.path.splitext(onnx_filename)
shaped_filename = "{name}_static{ext}".format(name=name, ext=ext)
print("Saving static input onnx to: {}".format(shaped_filename))
onnx.checker.check_model(tmp_net)
onnx.save(tmp_net, shaped_filename)