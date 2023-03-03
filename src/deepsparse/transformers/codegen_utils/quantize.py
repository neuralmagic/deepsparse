### WILL BE REMOVED LATER ###
import argparse
import os
from collections import defaultdict
import numpy
import onnx
from onnx.external_data_helper import convert_model_to_external_data

from onnxsim import simplify
from sparseml.exporters.onnx_to_deepsparse import ONNXToDeepsparse
from sparsifyml.one_shot import sparsify_fast


def run_transforms(transforms, model):
    for transform_class in transforms:
        print(f"running transform {transform_class.__name__}")
        transform = transform_class()
        model = transform.transform(model)

        model.graph.node.extend(transform._nodes_to_add)
        for node in transform._nodes_to_delete:
            model.graph.node.remove(node)

    return model


def _matmul_counts(model):
    counts = defaultdict(int)

    for node in model.graph.node:
        if not ("MatMul" in node.op_type or "Gemm" in node.op_type):
            continue
        counts[node.op_type] += 1
    return dict(counts)


def main(args):

    model = onnx.load(args.model)

    sparsify_fast(
        model=args.model,
        sparsity=None,
        quantization=dict(
            ignore=[
                node.op_type
                for node in model.graph.node
                if node.op_type not in ["MatMul", "Gemm"]
            ]
        ),
        sample_input=[[numpy.random.randint(0, 1, (384)), numpy.ones((384))]],
        save_path=os.path.join(args.save_dir, "model_quant.onnx"),
    )

    model = onnx.load(os.path.join(args.save_dir, "model_quant.onnx"))
    print(f"matmul op counts pre transform: {_matmul_counts(model)}")
    e = ONNXToDeepsparse()
    model = e.apply(model)

    print(f"matmul op counts post transform: {_matmul_counts(model)}")

    model, _ = simplify(model)
    convert_model_to_external_data(model, all_tensors_to_one_file=False)
    onnx.save(model, args.save_dir + "/model_quant_final.onnx")


if __name__ == "__main__":
    _PARSER = argparse.ArgumentParser()
    _PARSER.add_argument(
        "--model",
        required=True,
        help="ONNX model to quantize",
    )
    _PARSER.add_argument(
        "--save-dir", default="./quant_onnx", help="Directory to save models to"
    )
    main(_PARSER.parse_args())
