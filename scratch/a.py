import onnx
from onnx import numpy_helper
import torch
import numpy

from typing import Dict, Optional

from functools import lru_cache




def _onnx():
    import onnx

    # # Specify the path to the ONNX file
    onnx_path = "/home/ubuntu/george/nm/deepsparse/scratch/model.onnx"


    onnx_model = onnx.load(onnx_path)

    onnx_weight_names = [
        node.input[1]
        for node in onnx_model.graph.node
        if node.op_type in ["Conv", "MatMul", "Gemm", "MatMulInteger", "ConvInteger"]
    ]
    onnx_weight_names.extend([
        node.input[3]
        for node in onnx_model.graph.node
        if node.op_type in ["QLinearConv", "QLinearMatMul"]
    ])

    onnx_names_to_weights = {}
    hash_to_onnx_names = {}
    for init in onnx_model.graph.initializer:
        if init.name in onnx_weight_names:
            onnx_names_to_weights[init.name] =  numpy_helper.to_array(init)
            hash_to_onnx_names[hash(numpy_helper.to_array(init).tostring())] = init.name
            print(numpy.max(numpy_helper.to_array(init)))

    return onnx_names_to_weights, hash_to_onnx_names


    # weights_to_onnx_names = {}
    # for init in onnx_model.graph.initializer:
    #     if init.name in onnx_weight_names:
    #         weights_to_onnx_names[numpy_helper.to_array(init)] = init.name

    # assert len(onnx_names_to_weights) == len(onnx_weight_names)



# --- TORCH --- 
val, hash_to_onnx_names = _onnx()
print()
print()
print()
print()
print()
print()

# print(hash_to_onnx_names)

def _torch():
    import torch
    torch.set_printoptions(sci_mode=False)
    torch_path = "/home/ubuntu/george/nm/deepsparse/scratch/model.pth"

    torch_model = torch.load(torch_path, map_location=torch.device('cpu'))

    onnx_to_torch = {}
    c = 0
    onnx_to_pytorch_names = {}
    breakpoint()
    for key, val in torch_model["state_dict"].items():
        val = val.numpy()
        hashed_val = hash(val.tostring())
        print(numpy.max(val))
        if hashed_val in hash_to_onnx_names:
            onnx_name = hash_to_onnx_names[hashed_val] 
            onnx_to_torch[onnx_name] = key
            # breakpoint()


    # assert c == len(onnx_names_to_weights)
    print()
    print()
    print()
    print()
    print()
    print(onnx_to_torch)


_torch()


"""

torch_names_to_weights = {name: param for name, param in torch_model.named_paramters()
from onnx import numpy_helper
onnx_names_to_weights = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}
"""