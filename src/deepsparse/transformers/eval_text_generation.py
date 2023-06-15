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

import argparse
import os
from typing import List

import numpy as np
import onnxruntime
from transformers import AutoConfig, AutoTokenizer

from deepsparse.transformers.metrics import Perplexity
from evaluate import load


input_text1 = "While the Indiana Jones movies likely inspired a lot of young viewers to become interested in a career in archeology, they are not very realistic depictions of the profession. The movies seem very willing to admit this with some of the more poignant Indiana Jones quotes. "
input_text2 = "During an early scene in Indiana Jones and the Last Crusade, Indy lectures his students that a big part of being an archeologist is research. Of course, he then goes off on a thrilling adventure of treasure hunting where X does, in fact, mark the spot."


def perplexity_eval(args: argparse.Namespace):
    # not using the pipeline for now, since it does not support batched inputs
    session = onnxruntime.InferenceSession(os.path.join(args.model_path, "model.onnx"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)

    perplexity = Perplexity(
        session=session,
        tokenizer=tokenizer,
        vocab_size=config.vocab_size,
        static_length=args.sequence_length,
    )
    for input_text in args.dataset:
        perplexity.add_batch(input_text)

    return perplexity.compute()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate a text-generation model on a toy dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/damian/sparseml/deployment",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--dataset",
        type=List[str],
        default=[input_text1, input_text2],
        help="A list of strings to evaluate perplexity on",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length to use for evaluation",
    )
    args = parser.parse_args()
    results = perplexity_eval(args)

    # testing the correctness
    perplexity = load("perplexity", module_type="metric")
    gt_results = perplexity.compute(
        predictions=[input_text1, input_text2], model_id="facebook/opt-350M"
    )
    assert np.allclose(
        np.array(results["perplexities"]), np.array(gt_results["perplexities"])
    )
