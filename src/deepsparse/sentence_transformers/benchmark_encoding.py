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

import random
import string
import time

import sentence_transformers
from deepsparse.sentence_transformers import (
    SentenceTransformer as DeepSparseSentenceTransformer,
)


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


# Generate a list of random sentences
num_sentences = 100
sentences = [generate_random_sentence() for _ in range(num_sentences)]

# Model names
model_name = "BAAI/bge-small-en-v1.5"
sparse_model_name = "zeroshot/bge-small-en-v1.5-quant"

# Load the models
standard_model = sentence_transformers.SentenceTransformer(model_name)
deepsparse_model = DeepSparseSentenceTransformer(model_name, export=True)
deepsparse_opt_model = DeepSparseSentenceTransformer(sparse_model_name)

# Benchmark sentence_transformers
standard_time = benchmark_model(standard_model, sentences)
print(
    f"[Standard SentenceTransformer] Encoded {num_sentences} sentences "
    f"of length {len(sentences[0])} in {standard_time:.2f} seconds."
)

# Benchmark deepsparse.sentence_transformers
deepsparse_time = benchmark_model(deepsparse_model, sentences)
print(
    f"[DeepSparse] Encoded {num_sentences} sentences of length "
    f"{len(sentences[0])} in {deepsparse_time:.2f} seconds."
)

# Benchmark deepsparse.sentence_transformers
deepsparse_opt_time = benchmark_model(deepsparse_opt_model, sentences)
print(
    f"[DeepSparse Optimized] Encoded {num_sentences} sentences of length "
    f"{len(sentences[0])} in {deepsparse_opt_time:.2f} seconds."
)
