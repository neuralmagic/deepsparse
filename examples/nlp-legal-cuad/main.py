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

import datasets
from deepsparse import Pipeline


parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

pipeline = Pipeline.create(
    task="question-answering",
    model_path=args.model,
    version_2_with_negative=True,
)

cuad = datasets.load_dataset("cuad")["test"]

example = cuad[0]

output = pipeline(question=example["question"], context=example["context"])

print("Answered:", output.answer)
print("Expected:", example["answers"])
