import argparse

import numpy

from deepsparse import compile_model, cpu
from deepsparse.utils import verify_outputs
from sparsezoo.models import classification
from sparsezoo.objects import Model


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()

model_registry = {
    "mobilenet_v1": classification.mobilenet_v1,
    "mobilenet_v2": classification.mobilenet_v2,
    "resnet_18": classification.resnet_18,
    "resnet_34": classification.resnet_34,
    "resnet_50": classification.resnet_50,
    "resnet_101": classification.resnet_101,
    "resnet_152": classification.resnet_152,
    "efficientnet_b0": classification.efficientnet_b0,
    "efficientnet_b4": classification.efficientnet_b4,
}


def fetch_model(model_name: str) -> Model:
    if model_name not in model_registry:
        raise Exception(
            f"Could not find model '{model_name}' in classification model registry."
        )
    return model_registry[model_name]()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and check accuracy of image classification models"
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


def calculate_top1_accuracy(pred: numpy.array, labels: numpy.array) -> float:
    """
    :param pred: the model's prediction to compare with
    :param labels: the labels for the data to compare to
    :return: the calculated top1 accuracy
    """
    batch_size = pred.shape[0]
    pred = numpy.argmax(pred, axis=-1)

    correct = (pred == labels.reshape(pred.shape)).sum()
    correct *= 100.0 / batch_size

    return correct


def main():
    args = parse_args()
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

    # Measure accuracy against ground truth labels
    predicted_labels = numpy.argmax(predicted_outputs[-1], axis=-1)
    accuracy = calculate_top1_accuracy(predicted_outputs[-1], batched_labels[0])
    print("Top-1 Accuracy for batch size {}: {:.2f}%".format(batch_size, accuracy))

    # BENCHMARK
    # Record output from executing through the DeepSparse engine
    print("Benchmarking...")
    results = engine.benchmark(batched_inputs)
    print(results)


if __name__ == "__main__":
    main()
