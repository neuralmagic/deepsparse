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
import json
from cProfile import Profile
from pstats import Stats

from tqdm.auto import tqdm

from deepsparse import Pipeline


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
    question_answer = Pipeline.create(
        task="question-answering",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
        max_answer_length=args.max_answer_length,
        n_best_size=args.n_best_size,
        pad_to_max_length=args.pad_to_max_length,
        output_dir=args.output_dir,
    )
    print(f"Engine info: {question_answer.engine}")
    for idx, sample in _enumerate_progress(squad, args.max_samples):
        pred = question_answer(
            id=sample["id"],
            question=sample["question"],
            context=sample["context"],
        )

        squad_metrics.add_batch(
            predictions=[{"prediction_text": pred.answer, "id": sample["id"]}],
            references=[{"answers": sample["answers"], "id": sample["id"]}],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return squad_metrics


def mnli_eval(args):
    # load mnli validation dataset and eval tool
    mnli = load_dataset("glue", "mnli")
    mnli_matched = mnli["validation_matched"]
    mnli_mismatched = mnli["validation_mismatched"]
    mnli_metrics = load_metric("glue", "mnli")

    # load pipeline
    text_classify = Pipeline.create(
        task="text-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
    )
    print(f"Engine info: {text_classify.engine}")

    try:
        label_map = _get_label2id(text_classify.config_path)
    except KeyError:
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    for idx, sample in _enumerate_progress(mnli_matched, args.max_samples):
        pred = text_classify([[sample["premise"], sample["hypothesis"]]])
        mnli_metrics.add_batch(
            predictions=[label_map.get(pred.labels[0])],
            references=[sample["label"]],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    for idx, sample in _enumerate_progress(mnli_mismatched, args.max_samples):
        pred = text_classify([[sample["premise"], sample["hypothesis"]]])
        mnli_metrics.add_batch(
            predictions=[label_map.get(pred.labels[0])],
            references=[sample["label"]],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return mnli_metrics


def qqp_eval(args):
    # load qqp validation dataset and eval tool
    qqp = load_dataset("glue", "qqp")["validation"]
    qqp_metrics = load_metric("glue", "qqp")

    # load pipeline
    text_classify = Pipeline.create(
        task="text-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
    )
    print(f"Engine info: {text_classify.engine}")

    try:
        label_map = _get_label2id(text_classify.config_path)
    except KeyError:
        label_map = {"not_duplicate": 0, "duplicate": 1, "LABEL_0": 0, "LABEL_1": 1}

    for idx, sample in _enumerate_progress(qqp, args.max_samples):
        pred = text_classify([[sample["question1"], sample["question2"]]])

        qqp_metrics.add_batch(
            predictions=[label_map.get(pred.labels[0])],
            references=[sample["label"]],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return qqp_metrics


def sst2_eval(args):
    # load sst2 validation dataset and eval tool
    sst2 = load_dataset("glue", "sst2")["validation"]
    sst2_metrics = load_metric("glue", "sst2")

    # load pipeline
    text_classify = Pipeline.create(
        task="text-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
    )
    print(f"Engine info: {text_classify.engine}")

    try:
        label_map = _get_label2id(text_classify.config_path)
    except KeyError:
        label_map = {"negative": 0, "positive": 1, "LABEL_0": 0, "LABEL_1": 1}

    for idx, sample in _enumerate_progress(sst2, args.max_samples):
        pred = text_classify(
            sample["sentence"],
        )

        sst2_metrics.add_batch(
            predictions=[label_map.get(pred.labels[0])],
            references=[sample["label"]],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return sst2_metrics


def conll2003_eval(args):
    # load qqp validation dataset and eval tool
    conll2003 = load_dataset("conll2003")["validation"]
    conll2003_metrics = load_metric("seqeval")

    # load pipeline
    token_classify = Pipeline.create(
        task="token-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
    )
    print(f"Engine info: {token_classify.engine}")

    ner_tag_map = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }
    # map entity id and raw id from pipeline to NER tag
    label_map = {label_id: ner_tag for ner_tag, label_id in ner_tag_map.items()}
    label_map.update(
        {
            token_classify.config.id2label[label_id]: tag
            for tag, label_id in ner_tag_map.items()
        }
    )

    for idx, sample in _enumerate_progress(conll2003, args.max_samples):
        if not sample["tokens"]:
            continue  # invalid dataset item, no tokens
        pred = token_classify(inputs=sample["tokens"], is_split_into_words=True)
        pred_ids = [label_map[prediction.entity] for prediction in pred.predictions[0]]
        label_ids = [label_map[ner_tag] for ner_tag in sample["ner_tags"]]

        conll2003_metrics.add_batch(
            predictions=[pred_ids],
            references=[label_ids],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return conll2003_metrics


def _enumerate_progress(dataset, max_steps):
    progress_bar = tqdm(dataset, total=max_steps) if max_steps else tqdm(dataset)
    return enumerate(progress_bar)


def _get_label2id(config_file_path):
    with open(config_file_path) as f:
        config = json.load(f)
    return config["label2id"]


# Register all the supported downstream datasets here
SUPPORTED_DATASETS = {
    "squad": squad_eval,
    "mnli": mnli_eval,
    "qqp": qqp_eval,
    "sst2": sst2_eval,
    "conll2003": conll2003_eval,
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

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=("Folder to save output predictions, used for debugging"),
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=("Run with profiling, used for debugging"),
    )

    # Arguments specific for the Question Answering task
    parser.add_argument(
        "--max-answer-length",
        help="The maximum length of an answer that can be generated. This is "
        "needed because the start and end predictions are not conditioned "
        "on one another.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--version-2-with-negative",
        help="Whether or not the underlying dataset contains examples with "
        "no answers",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--pad-to-max-length",
        help="Whether to pad all samples to `max_seq_length`. If False, "
        "will pad the samples dynamically when batching to the maximum length "
        "in the batch (which can be faster on GPU but will be slower on TPU).",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--n-best-size",
        help="The total number of n-best predictions to generate when looking "
        "for an answer.",
        type=int,
        default=20,
    )

    return parser.parse_args()


def _main(args):
    dataset = args.dataset.lower()

    if dataset not in SUPPORTED_DATASETS:
        raise KeyError(
            f"Unknown downstream dataset {args.dataset}, "
            f"available datasets are {list(SUPPORTED_DATASETS.keys())}"
        )

    metrics = SUPPORTED_DATASETS[dataset](args)

    print(f"\n{dataset} eval results: {metrics.compute()}")


def main():
    args = parse_args()
    if args.profile:
        profiler = Profile()
        profiler.runcall(lambda: _main(args))
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()
    else:
        _main(args)


if __name__ == "__main__":
    main()
