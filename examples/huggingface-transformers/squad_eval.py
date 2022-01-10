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
Example script for evaluating an ONNX model on the SQuAD dataset using the
DeepSparse Engine or onnxruntime

##########
Command help:
usage: squad_eval.py [-h] [-c NUM_CORES] [-e {deepsparse,onnxruntime}]
                     [--max-sequence-length MAX_SEQUENCE_LENGTH]
                     onnx_filepath

Evaluate a BERT ONNX model on the SQuAD dataset

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file or SparseZoo
                        stub to the model

optional arguments:
  -h, --help            show this help message and exit
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the eval on,
                        defaults to all physical cores available on the system
  -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                        Inference engine backend to run eval on. Choices are
                        'deepsparse', 'onnxruntime'. Default is 'deepsparse'
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        the max sequence length for model inputs. Default is
                        384

##########
Example command for evaluating a sparse BERT QA model from sparsezoo:
python squad_eval.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98
"""


import argparse

from tqdm.auto import tqdm

from datasets import load_dataset, load_metric
from deepsparse.transformers import pipeline


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a BERT ONNX model on the SQuAD dataset"
    )
    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub to the model",
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

    return parser.parse_args()


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

    for sample in tqdm(squad):
        pred = question_answer(
            question=sample["question"],
            context=sample["context"],
            num_spans=1,  # only look at first part of long contexts
        )

        squad_metrics.add_batch(
            predictions=[{"prediction_text": pred["answer"], "id": sample["id"]}],
            references=[{"answers": sample["answers"], "id": sample["id"]}],
        )

    print(f"\nSQuAD eval results: {squad_metrics.compute()}")


def main():
    args = parse_args()
    squad_eval(args)


if __name__ == "__main__":
    main()
