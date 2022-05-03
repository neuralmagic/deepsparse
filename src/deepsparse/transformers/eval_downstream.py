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
Example script for evaluating an ONNX model on a downstream dataset using the
DeepSparse Engine or ONNXRuntime

##########
Command help:
usage: eval_downstream.py [-h] [-d {squad,mnli,qqp,sst2}] [-c NUM_CORES]
                          [-e {deepsparse,onnxruntime}]
                          [--max-sequence-length MAX_SEQUENCE_LENGTH]
                          [--max-samples MAX_SAMPLES]
                          onnx_filepath

Evaluate a BERT ONNX model on a downstream dataset

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file or SparseZoo
                        stub to the model

optional arguments:
  -h, --help            show this help message and exit
  -d {squad,mnli,qqp,sst2}, --dataset {squad,mnli,qqp,sst2}
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the eval on,
                        defaults to all physical cores available on the system
  -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                        Inference engine backend to run eval on. Choices are
                        'deepsparse', 'onnxruntime'. Default is 'deepsparse'
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        the max sequence length for model inputs. Default is
                        384
  --max-samples MAX_SAMPLES
                        the max number of samples to evaluate. Default is None
                        or all samples

##########
Example command for evaluating a sparse BERT QA model from sparsezoo:
python eval_downstream.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none \
    --dataset squad
"""


import argparse

from tqdm.auto import tqdm

from deepsparse.transformers import pipeline


from datasets import load_dataset, load_metric  # isort: skip


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SQUAD_KEY = "squad"
MNLI_KEY = "mnli"
QQP_KEY = "qqp"
SST2_KEY = "sst2"


def squad_eval(args):
    # load squad validation dataset and eval tool
    squad = load_dataset("squad")["validation"]
    squad_metrics = load_metric("squad")

    # load QA pipeline
    question_answer = pipeline(
        task="question-answering",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        max_length=args.max_sequence_length,
    )
    print(f"Engine info: {question_answer.model}")

    for idx, sample in enumerate(tqdm(squad)):
        pred = question_answer(
            question=sample["question"],
            context=sample["context"],
            num_spans=1,  # only look at first part of long contexts
        )

        squad_metrics.add_batch(
            predictions=[{"prediction_text": pred["answer"], "id": sample["id"]}],
            references=[{"answers": sample["answers"], "id": sample["id"]}],
        )

        if args.max_samples and idx > args.max_samples:
            break

    return squad_metrics


def mnli_eval(args):
    # load mnli validation dataset and eval tool
    mnli = load_dataset("glue", "mnli")
    mnli_matched = mnli["validation_matched"]
    mnli_mismatched = mnli["validation_mismatched"]
    mnli_metrics = load_metric("glue", "mnli")

    # load pipeline
    text_classify = pipeline(
        task="text-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        max_length=args.max_sequence_length,
    )
    print(f"Engine info: {text_classify.model}")

    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    for idx, sample in enumerate(tqdm(mnli_matched)):
        pred = text_classify([[sample["premise"], sample["hypothesis"]]])
        mnli_metrics.add_batch(
            predictions=[label_map.get(pred[0]["label"])],
            references=[sample["label"]],
        )

        if args.max_samples and idx > args.max_samples:
            break

    for idx, sample in enumerate(tqdm(mnli_mismatched)):
        pred = text_classify([[sample["premise"], sample["hypothesis"]]])
        mnli_metrics.add_batch(
            predictions=[label_map.get(pred[0]["label"])],
            references=[sample["label"]],
        )

        if args.max_samples and idx > args.max_samples:
            break

    return mnli_metrics


def qqp_eval(args):
    # load qqp validation dataset and eval tool
    qqp = load_dataset("glue", "qqp")["validation"]
    qqp_metrics = load_metric("glue", "qqp")

    # load pipeline
    text_classify = pipeline(
        task="text-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        max_length=args.max_sequence_length,
    )
    print(f"Engine info: {text_classify.model}")

    label_map = {"not_duplicate": 0, "duplicate": 1}

    for idx, sample in enumerate(tqdm(qqp)):
        pred = text_classify([[sample["question1"], sample["question2"]]])

        qqp_metrics.add_batch(
            predictions=[label_map.get(pred[0]["label"])],
            references=[sample["label"]],
        )

        if args.max_samples and idx > args.max_samples:
            break

    return qqp_metrics


def sst2_eval(args):
    # load sst2 validation dataset and eval tool
    sst2 = load_dataset("glue", "sst2")["validation"]
    sst2_metrics = load_metric("glue", "sst2")

    # load pipeline
    text_classify = pipeline(
        task="text-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        max_length=args.max_sequence_length,
    )
    print(f"Engine info: {text_classify.model}")

    label_map = {"negative": 0, "positive": 1}

    for idx, sample in enumerate(tqdm(sst2)):
        pred = text_classify(
            sample["sentence"],
        )

        sst2_metrics.add_batch(
            predictions=[label_map.get(pred[0]["label"])],
            references=[sample["label"]],
        )

        if args.max_samples and idx > args.max_samples:
            break

    return sst2_metrics


# Register all the supported downstream datasets here
SUPPORTED_DATASETS = {
    "squad": squad_eval,
    "mnli": mnli_eval,
    "qqp": qqp_eval,
    "sst2": sst2_eval,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a BERT ONNX model on a downstream dataset"
    )
    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub to the model",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(SUPPORTED_DATASETS.keys()),
        required=True,
    )

    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=0,
        help=(
            "The number of physical cores to run the eval on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[DEEPSPARSE_ENGINE, ORT_ENGINE],
        help=(
            "Inference engine backend to run eval on. Choices are 'deepsparse', "
            "'onnxruntime'. Default is 'deepsparse'"
        ),
    )
    parser.add_argument(
        "--max-sequence-length",
        help="the max sequence length for model inputs. Default is 384",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--max-samples",
        help="the max number of samples to evaluate. Default is None or all samples",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset.lower()

    if dataset not in SUPPORTED_DATASETS:
        raise KeyError(
            f"Unknown downstream dataset {args.dataset}, "
            f"available datasets are {list(SUPPORTED_DATASETS.keys())}"
        )

    metrics = SUPPORTED_DATASETS[dataset](args)

    print(f"\n{dataset} eval results: {metrics.compute()}")


if __name__ == "__main__":
    main()
