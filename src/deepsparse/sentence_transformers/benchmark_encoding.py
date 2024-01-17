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
from time import perf_counter

from deepsparse.sentence_transformers import DeepSparseSentenceTransformer
from sentence_transformers import SentenceTransformer


def generate_random_sentence(length):
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


def benchmark_model(model, sentences, batch_size):
    # Benchmark the encoding time for a model with a given list of sentences.
    start_time = perf_counter()
    _ = model.encode(sentences, batch_size=batch_size)
    elapsed_time = perf_counter() - start_time
    return elapsed_time


def main(args):
    # Generate a list of random sentences
    sentences = [
        generate_random_sentence(args.length) for _ in range(args.num_sentences)
    ]

    # Load the models
    standard_model = SentenceTransformer(args.base_model, device="cpu")
    deepsparse_model = DeepSparseSentenceTransformer(args.sparse_model)

    standard_latency = benchmark_model(standard_model, sentences, args.batch_size)
    standard_throughput = len(sentences) / standard_latency * args.batch_size
    print(
        f"\n[SentenceTransformer]\n"
        f"Batch size: {args.batch_size}, Sentence length: {args.length}\n"
        f"Latency: {args.num_sentences} sentences in {standard_latency:.2f} seconds\n"
        f"Throughput: {standard_throughput:.2f} sentences/second"
    )

    deepsparse_latency = benchmark_model(deepsparse_model, sentences, args.batch_size)
    deepsparse_throughput = len(sentences) / deepsparse_latency * args.batch_size
    print(
        f"\n[DeepSparse Optimized]\n"
        f"Batch size: {args.batch_size}, Sentence length: {args.length}\n"
        f"Latency: {args.num_sentences} sentences in {deepsparse_latency:.2f} seconds\n"
        f"Throughput: {deepsparse_throughput:.2f} sentences/second"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Sentence Transformer Models for Latency and Throughput."
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for model inference.",
    )
    args = parser.parse_args()

    main(args)
