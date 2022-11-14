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
usage: eval_downstream.py [-h] [-d {squad,mnli,qqp,sst2,imdb,conll2003}]
                          [-c NUM_CORES]
                          [-e {deepsparse,onnxruntime}]
                          [--max-sequence-length MAX_SEQUENCE_LENGTH]
                          [--max-samples MAX_SAMPLES] [--zero-shot BOOL]
                          onnx_filepath

Evaluate a BERT ONNX model on a downstream dataset

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file or SparseZoo
                        stub to the model

optional arguments:
  -h, --help            show this help message and exit
  -d {squad,mnli,qqp,sst2,imdb,conll2003}, --dataset {squad,mnli,qqp,sst2,
                        imdb,conll2003}
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the eval on,
                        defaults to all physical cores available on the system
  -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                        Inference engine backend to run eval on. Choices are
                        'deepsparse', 'onnxruntime'. Default is 'deepsparse'
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        The max sequence length for model inputs. Default is
                        384
  --max-samples MAX_SAMPLES
                        The max number of samples to evaluate. Default is None
                        or all samples
  --zero-shot BOOL
                        Whether to run the dataset with a zero shot pipeline.
                        Currently supports zero shot pipelines for sst2.
                        Default is False

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

import numpy
from tqdm.auto import tqdm

from deepsparse import Pipeline
from deepsparse.transformers.metrics import PrecisionRecallF1


from datasets import load_dataset, load_metric  # isort: skip

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


def qa_eval(args, dataset_name="squad"):
    # load validation dataset and eval tool
    dataset = load_dataset(dataset_name)["validation"]
    qa_metrics = load_metric(dataset_name)

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
        version_2_with_negative=dataset_name == "squad_v2",
    )
    print(f"Engine info: {question_answer.engine}")
    for idx, sample in _enumerate_progress(dataset, args.max_samples):
        pred = question_answer(
            id=sample["id"],
            question=sample["question"],
            context=sample["context"],
        )

        predictions = [{"prediction_text": pred.answer, "id": sample["id"]}]
        if question_answer.version_2_with_negative:
            predictions[0]["no_answer_probability"] = 0.0

        qa_metrics.add_batch(
            predictions=predictions,
            references=[{"answers": sample["answers"], "id": sample["id"]}],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return qa_metrics


def mnli_eval(args):
    # load mnli validation dataset and eval tool
    mnli = load_dataset("glue", "mnli")
    mnli_matched = mnli["validation_matched"]
    mnli_mismatched = mnli["validation_mismatched"]
    mnli_metrics_matched = load_metric("glue", "mnli")
    mnli_metrics_mismatched = load_metric("glue", "mnli")

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
        mnli_metrics_matched.add_batch(
            predictions=[label_map.get(pred.labels[0])],
            references=[sample["label"]],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    for idx, sample in _enumerate_progress(mnli_mismatched, args.max_samples):
        pred = text_classify([[sample["premise"], sample["hypothesis"]]])
        mnli_metrics_mismatched.add_batch(
            predictions=[label_map.get(pred.labels[0])],
            references=[sample["label"]],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return mnli_metrics_matched, mnli_metrics_mismatched


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


def sst2_zero_shot_eval(args):
    # load sst2 validation dataset and eval tool
    sst2 = load_dataset("glue", "sst2")["validation"]
    sst2_metrics = load_metric("glue", "sst2")

    # load pipeline
    text_classify = Pipeline.create(
        task="zero_shot_text_classification",
        batch_size=2,
        model_scheme="mnli",
        model_config={
            "hypothesis_template": "The sentiment of this text is {}",
            "multi_class": True,
        },
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
        labels=["positive", "negative"],
    )
    print(f"Engine info: {text_classify.engine}")

    label_map = {"positive": 1, "negative": 0}

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


def imdb_eval(args):
    # load IMDB test dataset and eval tool
    imdb = load_dataset("imdb")
    if args.val_ratio is not None:
        _, imdb = _split_train_val(
            imdb["train"], args.val_ratio, seed=args.val_split_seed
        )
    else:
        imdb = imdb["test"]
    imdb_metrics = load_metric("accuracy")

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
        raise KeyError("label2id not found in model config")

    for idx, sample in _enumerate_progress(imdb, args.max_samples):
        pred = text_classify([sample["text"]])

        imdb_metrics.add_batch(
            predictions=[label_map.get(pred.labels[0])],
            references=[sample["label"]],
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return imdb_metrics


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


def go_emotions_eval(args):
    # load go_emotions validation dataset and eval tool
    go_emotions = load_dataset("go_emotions")["validation"]
    num_labels = 28

    # load pipeline
    text_classify = Pipeline.create(
        task="text-classification",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
        top_k=num_labels,
    )
    print(f"Engine info: {text_classify.engine}")

    go_emotions_metrics = PrecisionRecallF1(id_to_label=text_classify.config.id2label)
    ordered_labels = [text_classify.config.id2label[idx] for idx in range(num_labels)]

    for idx, sample in _enumerate_progress(go_emotions, args.max_samples):
        pred = text_classify(sample["text"])

        # order scores by label index and threshold
        label_to_scores = dict(zip(pred.labels[0], pred.scores[0]))
        predictions = numpy.array([label_to_scores[label] for label in ordered_labels])
        predictions = (predictions > 0.3).astype(int)  # threshold used in paper

        # one hot encode targets
        targets = numpy.zeros(num_labels)
        targets[sample["labels"]] = 1

        go_emotions_metrics.add_batch(
            predictions=predictions,
            targets=targets,
        )

        if args.max_samples and idx >= args.max_samples:
            break

    return go_emotions_metrics


def _enumerate_progress(dataset, max_steps):
    progress_bar = tqdm(dataset, total=max_steps) if max_steps else tqdm(dataset)
    return enumerate(progress_bar)


def _get_label2id(config_file_path):
    with open(config_file_path) as f:
        config = json.load(f)
    return config["label2id"]


def _split_train_val(train_dataset, val_ratio, seed=42):
    # Fixed random seed to make split consistent across runs with the same ratio
    ds = train_dataset.train_test_split(
        test_size=val_ratio, stratify_by_column="label", seed=seed
    )
    train_ds = ds.pop("train")
    val_ds = ds.pop("test")
    return train_ds, val_ds


# Register all the supported downstream datasets here
SUPPORTED_DATASETS = {
    "squad": lambda args: qa_eval(args, dataset_name="squad"),
    "squad_v2": lambda args: qa_eval(args, dataset_name="squad_v2"),
    "mnli": mnli_eval,
    "qqp": qqp_eval,
    "sst2": sst2_eval,
    "sst2_zero_shot": sst2_zero_shot_eval,
    "imdb": imdb_eval,
    "conll2003": conll2003_eval,
    "go_emotions": go_emotions_eval,
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
        "-v",
        "--val_ratio",
        type=float,
        default=None,
        help=(
            "Ratio between 0.0 and 1.0 representing the proportion "
            "of the dataset include in the validation set"
        ),
    )
    parser.add_argument(
        "-s",
        "--val_split_seed",
        type=int,
        default=42,
        help=(
            "Random seed used to split the validation set, used with "
            "the --val_ratio flag. Default to 42."
        ),
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
    parser.add_argument(
        "--zero-shot",
        help="Whether to run the dataset with a zero shot pipeline. Currently "
        "supports sst2. Default is False",
        type=bool,
        default=False,
    )

    return parser.parse_args()


def _main(args):
    dataset = args.dataset.lower()
    if args.zero_shot:
        dataset += "_zero_shot"

    if dataset not in SUPPORTED_DATASETS:
        raise KeyError(
            f"Unknown downstream dataset {args.dataset}, "
            f"available datasets are {list(SUPPORTED_DATASETS.keys())}"
        )

    if dataset == "mnli":
        mnli_metrics_matched, mnli_metrics_mismatched = mnli_eval(args)
        mnli_metrics_matched = mnli_metrics_matched.compute()
        mnli_metrics_mismatched = mnli_metrics_mismatched.compute()
        mnli_metrics = {k + "_m": v for k, v in mnli_metrics_matched.items()}
        mnli_metrics.update({k + "_mm": v for k, v in mnli_metrics_mismatched.items()})
        print(f"\nmnli eval results: {mnli_metrics}")
    else:
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
