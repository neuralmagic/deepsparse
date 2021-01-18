import argparse

import numpy

from deepsparse import compile_model, cpu
from deepsparse.utils import verify_outputs
from sparsezoo import Model
from sparsezoo.models import detection


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()

model_registry = {
    "ssd_resnet50_300": detection.ssd_resnet50_300,
    "yolo_v3": detection.yolo_v3,
}


def fetch_model(model_name: str) -> Model:
    if model_name not in model_registry:
        raise Exception(
            f"Could not find model '{model_name}' in detection model registry."
        )
    return model_registry[model_name]()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and check accuracy of object detection models"
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=model_registry.keys(),
        help="Model type to analyze",
    )

    parser.add_argument(
        "-s",
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-j",
        "--num_cores",
        type=int,
        default=CORES_PER_SOCKET,
        help="The number of physical cores to run the analysis on, "
        "defaults to all physical cores available on the system",
    )

    return parser.parse_args()


def main(args):
    model = fetch_model(args.model_name)
    batch_size = args.batch_size
    num_cores = args.num_cores

    # Gather batch of data
    batch = model.sample_batch(batch_size=batch_size)
    batched_inputs = batch["inputs"]
    batched_outputs = batch["outputs"]
    batched_labels = batch["labels"]

    # Compile model for inference
    print("Compiling {} model with DeepSparse Engine".format(model.architecture_id))
    engine = compile_model(model, batch_size, num_cores)
    print(engine)

    # INFERENCE
    # Record output from inference through the DeepSparse Engine
    print("Executing...")
    predicted_outputs = engine(batched_inputs)

    # Compare against reference model output
    verify_outputs(predicted_outputs, batched_outputs)

    # BENCHMARK
    # Record output from executing through the DeepSparse engine
    print("Benchmarking...")
    results = engine.benchmark(batched_inputs)
    print(results)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
