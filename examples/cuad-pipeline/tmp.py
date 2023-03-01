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

from deepsparse.transformers import QuestionAnsweringPipeline
import datasets


pipeline = QuestionAnsweringPipeline(
    model_path="/nm/drive3/tuan/models/CUAD/sparse_transfer/potential/CUAD@oberta-pruned90@sparse_transfer_decay_cuad@EP10@BS32@H1.0@T5.0@WD0.0001@LR8e-5@ID20943/checkpoint-1536/deployment",
    version_2_with_negative=True,
    doc_stride=256,
    max_answer_length=512,
    max_question_length=512,
    sequence_length=512,
    num_cores=2,
)
cuad_test = datasets.load_dataset("cuad")["test"]

example = cuad_test[0]

output = pipeline(question=example["question"], context=example["context"])

print(example["question"])
print(output.answer)
print()
print("Correct answer:", example["answers"])
