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

import json

from deepsparse import Pipeline


pipeline = Pipeline.create(task="sentiment-analysis", model_path="./model/deployment")


def qa_lambda_handler(event, context):

    body = json.loads(event["body"])
    question = body["question"]
    context = body["context"]

    inference = pipeline(question=question, context=context)
    print(f"Question: {question}, Answer: {inference.answer}")

    return {
        "statusCode": 200,
        "body": json.dumps({"Question": question, "Answer": inference.answer}),
    }


def sa_lambda_handler(event, context):

    body = json.loads(event["body"])
    sequences = body["sequences"]
    print(sequences)

    inference = pipeline(sequences)
    print(f"Sentiment: {inference.labels}")

    return {
        "statusCode": 200,
        "body": json.dumps({"Answer": inference.labels}),
    }
