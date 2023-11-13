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
                          model_path

Evaluate a BERT ONNX model on a downstream dataset

positional arguments:
  model_path            The path to a directory containing model.onnx,
                        config.json, and tokenizer.json files or SparseZoo stub
                        to the model

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
import logging
from cProfile import Profile
from pstats import Stats

import numpy
from tqdm.auto import tqdm

from datasets import load_dataset, load_metric
from deepsparse import DEEPSPARSE_ENGINE, ORT_ENGINE, Pipeline
from deepsparse.transformers.metrics import Perplexity, PrecisionRecallF1
from deepsparse.transformers.utils.eval_helpers import process_concatenated_datasets


_LOGGER = logging.getLogger(__name__)


PPL_DATASETS = ["wikitext2", "c4", "openai_humaneval"]


def perplexity_eval(args, dataset_name="openai_humaneval"):
    if dataset_name in ["wikitext2", "c4"]:
        if args.kwargs is None:
            kwargs = {}
        else:
            kwargs = json.loads(args.kwargs)
        dataset = process_concatenated_datasets(
            dataset_name,
            args.model_path,
            args.max_sequence_length,
            kwargs,
        )
        # Set perplexity computation to accumulate negative log-likelihood across
        # sections
        accumulate = True
    else:
        dataset = load_dataset(dataset_name, split="test")
        accumulate = False

    # We'll use the text generation pipeline to generate a single token.
    # Along with the token, it returns the logits for input sequence
    text_generation = Pipeline.create(
        task="text-generation",
        model_path=args.model_path,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
        trust_remote_code=args.trust_remote_code,
    )

    # Instantiate perplexity metric
    perplexity_metrics = Perplexity(accumulate=accumulate)

    # Loop through samples
    batch_samples = []
    run_inference = False
    end_evaluation = False
    dataset_length = len(dataset)
    for idx, sample in _enumerate_progress(dataset, args.max_samples):
        # Collect input sequence
        if dataset_name == "openai_humaneval":
            sample = sample["prompt"] + sample["canonical_solution"]
        batch_samples.append(sample)

        if args.max_samples and idx == args.max_samples - 1:
            run_inference = True
            end_evaluation = True

        if (idx + 1) % args.batch_size == 0 or idx == dataset_length - 1:
            run_inference = True

        if run_inference:
            # Perform single token generation
            prediction = text_generation(
                sequences=batch_samples,
                output_scores=True,
                return_input_tokens=True,
                fixed_sequences_length=True,
                include_prompt_logits=True,
                max_length=1,
            )

            # Handle one sample at a time to make it simpler for masking
            for s in range(len(batch_samples)):
                # Need to remove tokens that were masked
                input_ids = prediction.input_tokens["input_ids"][s].flatten()
                logits = prediction.generations[s].score
                attention_mask = prediction.input_tokens["attention_mask"][s].flatten()

                effective_sequence_length = logits.shape[0]

                input_ids = input_ids[-effective_sequence_length:]
                attention_mask = attention_mask[-effective_sequence_length:]

                logits = numpy.compress(attention_mask, logits, axis=0)[:-1, :]
                input_ids = numpy.compress(attention_mask, input_ids)[1:]

                # Add predictions (logits) and targets (input_ids) to metric
                perplexity_metrics.add_batch(logits, input_ids)

            # Reset batch
            batch_samples.clear()
            run_inference = False

        if end_evaluation:
            break

    return perplexity_metrics


def qa_eval(args, dataset_name="squad"):
    # load validation dataset and eval tool
    dataset = load_dataset(dataset_name)["validation"]
    qa_metrics = load_metric(dataset_name)

    # load QA pipeline
    question_answer = Pipeline.create(
        task="question-answering",
        model_path=args.model_path,
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
        model_path=args.model_path,
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
        model_path=args.model_path,
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
        model_path=args.model_path,
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
        model_path=args.model_path,
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
        model_path=args.model_path,
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
        model_path=args.model_path,
        engine_type=args.engine,
        num_cores=args.num_cores,
        sequence_length=args.max_sequence_length,
        ignore_labels=[],
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
        model_path=args.model_path,
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
    "openai_humaneval": lambda args: perplexity_eval(
        args,
        dataset_name="openai_humaneval",
    ),
    "wikitext2": lambda args: perplexity_eval(
        args,
        dataset_name="wikitext2",
    ),
    "c4": lambda args: perplexity_eval(
        args,
        dataset_name="c4",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face Transformers "
        "ONNX model on a downstream dataset"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help=(
            "The path to a directory containing model.onnx, config.json, and "
            "tokenizer.json files or SparseZoo stub to the model"
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=list(SUPPORTED_DATASETS.keys()),
        required=True,
        type=str,
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
    parser.add_argument(
        "--batch-size",
        help="Batch size with which to evaluate model. Default is 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--trust-remote-code",
        help="Whether to allow for remote code execution in transformers.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--kwargs",
        help="Additional arguments specific to each dataset",
        type=str,
        default=None,
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

    if dataset not in PPL_DATASETS:
        _LOGGER.warning(
            "Batch-size argument is not supported for this dataset."
            "Will use default value of 1."
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
