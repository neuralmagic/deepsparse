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

from concurrent.futures import ThreadPoolExecutor
from deepsparse import Context
from deepsparse.transformers import QuestionAnsweringPipeline
import datasets
from tqdm import tqdm
from cuad_eval import CUAD

PATH = "/home/corey/cuad-quant-deployment"


def main():
    cuad = datasets.load_dataset(
        "cuad",
        split="test",
    )
    metric = CUAD()

    context = Context(num_cores=20, num_streams=20)
    executor = ThreadPoolExecutor(max_workers=context.num_streams)

    pipeline = QuestionAnsweringPipeline(
        model_path=PATH,
        version_2_with_negative=True,
        doc_stride=256,
        max_answer_length=512,
        max_question_length=512,
        sequence_length=512,
        context=context,
        executor=executor,
    )

    for i, example in tqdm(enumerate(cuad), total=len(cuad)):
        if len(example["context"]) > 100_000:
            continue
        output = pipeline(question=example["question"], context=example["context"])

        metric.add_batch(
            predictions=[
                {
                    "id": example["id"],
                    "prediction_text": [{"text": output.answer, "probability": 1.0}],
                }
            ],
            references=[{"id": example["id"], "answers": example["answers"]}],
        )

    print(metric.compute())


if __name__ == "__main__":
    main()
