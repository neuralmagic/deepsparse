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
Example script for an application that runs inferences on the SQuAD dataset using a
BERT-QA ONNX model and the DeepSparse engine

##########
Command help:
usage: squad_inference.py [-h] [-c NUM_CORES] [-e {deepsparse,onnxruntime}]
                          [-s NUM_SOCKETS]
                          [--max-sequence-length MAX_SEQUENCE_LENGTH]
                          [--num-samples NUM_SAMPLES]
                          [--display-frequency DISPLAY_FREQUENCY]
                          onnx_filepath

Run a BERT ONNX model on the SQuAD dataset

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file or SparseZoo
                        stub to the model

optional arguments:
  -h, --help            show this help message and exit
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the application
                        on, defaults to all physical cores available on the
                        system
  -e {deepsparse,onnxruntime}, --engine {deepsparse,onnxruntime}
                        Inference engine backend to run application on.
                        Choices are 'deepsparse', 'onnxruntime'. Default is
                        'deepsparse'
  -s NUM_SOCKETS, --num-sockets NUM_SOCKETS
                        For DeepSparse only. The number of physical cores to
                        run the benchmark on. Defaults to None where is uses
                        all sockets available on the system
  --max-sequence-length MAX_SEQUENCE_LENGTH
                        the max sequence length for model inputs. Default is
                        384
  --num-samples NUM_SAMPLES
                        number of samples to run inferences for, cannot be
                        greater than the number of examples in the validation
                        dataset. Default 0 runs the entire validation dataset
  --display-frequency DISPLAY_FREQUENCY
                        the question, context, and predicted answer will be
                        displayed every display_frequency samples. Default is
                        1

##########
Example command for running using a model from sparsezoo:
python server.py \
    zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-moderate
"""
import argparse

from tqdm.auto import tqdm

from datasets import load_dataset
from pipelines import pipeline


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a BERT ONNX model on the SQuAD dataset"
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
            "The number of physical cores to run the application on, "
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
            "Inference engine backend to run application on. Choices are 'deepsparse', "
            "'onnxruntime'. Default is 'deepsparse'"
        ),
    )
    parser.add_argument(
        "-s",
        "--num-sockets",
        type=int,
        default=None,
        help=(
            "For DeepSparse only. The number of physical cores to run the "
            "benchmark on. Defaults to None where is uses all sockets available on the "
            "system"
        ),
    )
    parser.add_argument(
        "--max-sequence-length",
        help="the max sequence length for model inputs. Default is 384",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--num-samples",
        help=(
            "number of samples to run inferences for, cannot be greater than the "
            "number of examples in the validation dataset. Default 0 runs the entire "
            "validation dataset"
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--display-frequency",
        help=(
            "the question, context, and predicted answer will be displayed every "
            "display_frequency samples. Default is 1"
        ),
        type=int,
        default=10,
    )

    return parser.parse_args()


def squad_inference(args):
    # load squad validation dataset
    squad = load_dataset("squad")["validation"]
    squad = squad.shuffle()

    num_samples = args.num_samples or len(squad)
    if num_samples > len(squad):
        raise ValueError(
            f"num_samples ({num_samples}) cannot be greater than the number of samples "
            f"in the validation dataset ({len(squad)})."
        )

    # load QA pipeline
    question_answer = pipeline(
        task="question-answering",
        model_path=args.onnx_filepath,
        engine_type=args.engine,
        num_cores=args.num_cores,
        num_sockets=args.num_sockets,
        max_length=args.max_sequence_length,
    )

    print(chr(27) + "[2J")  # clear screen
    for idx in tqdm(range(num_samples), position=0, leave=False):
        # run inference
        pred = question_answer(
            question=squad[idx]["question"],
            context=squad[idx]["context"],
            max_doc_strides=1,  # only look at first part of long contexts
        )

        # display every display_frequency samples
        if idx % args.display_frequency == 0:
            tqdm.write(
                f"\033[2J"  # clear previous output
                f"question: {_color_text(squad[idx]['question'], _TermColors.BLUE)}\n"
                f"answer: {_color_text(pred['answer'], _TermColors.RED)}"
            )


class _TermColors:
    RED = "\033[91m"
    BLUE = "\033[94m"
    END = "\033[0m"


def _color_text(text: str, color: str) -> str:
    return f"{color}{text}{_TermColors.END}"


def main():
    args = parse_args()
    squad_inference(args)


if __name__ == "__main__":
    main()
