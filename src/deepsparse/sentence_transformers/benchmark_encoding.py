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
import random
import string
import time

import sentence_transformers
from deepsparse.sentence_transformers import DeepSparseSentenceTransformer


def generate_random_sentence(length=700):
    # Generate a random sentence of a given length.
    return "".join(
        random.choices(
            string.ascii_letters
            + string.digits
            + string.punctuation
            + string.whitespace,
            k=length,
        )
    )


def benchmark_model(model, sentences):
    # Benchmark the encoding time for a model with a given list of sentences.
    start_time = time.time()
    _ = model.encode(sentences)
    elapsed_time = time.time() - start_time
    return elapsed_time


def main(args):
    # Generate a list of random sentences
    sentences = [
        generate_random_sentence(args.length) for _ in range(args.num_sentences)
    ]

    # Load the models
    standard_model = sentence_transformers.SentenceTransformer(args.base_model)
    deepsparse_model = DeepSparseSentenceTransformer(args.base_model, export=True)
    deepsparse_opt_model = DeepSparseSentenceTransformer(args.sparse_model)

    # Benchmark sentence_transformers
    standard_time = benchmark_model(standard_model, sentences)
    print(
        f"[Standard SentenceTransformer] Encoded {args.num_sentences} sentences "
        f"of length {args.length} in {standard_time:.2f} seconds."
    )

    # Benchmark deepsparse.sentence_transformers
    deepsparse_time = benchmark_model(deepsparse_model, sentences)
    print(
        f"[DeepSparse] Encoded {args.num_sentences} sentences of length "
        f"{args.length} in {deepsparse_time:.2f} seconds."
    )

    # Benchmark deepsparse.sentence_transformers
    deepsparse_opt_time = benchmark_model(deepsparse_opt_model, sentences)
    print(
        f"[DeepSparse Optimized]Encoded {args.num_sentences} sentences of length "
        f"{args.length} in {deepsparse_opt_time:.2f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Sentence Transformer Models."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Name of the standard model.",
    )
    parser.add_argument(
        "--sparse_model",
        type=str,
        default="zeroshot/bge-small-en-v1.5-quant",
        help="Name of the sparse model.",
    )
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=100,
        help="Number of sentences to generate.",
    )
    parser.add_argument(
        "--length", type=int, default=700, help="Length of each sentence."
    )
    args = parser.parse_args()

    main(args)
