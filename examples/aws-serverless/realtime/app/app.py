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
This Lambda function performs sentiment analysis on input data passed
in as a JSON payload using a pre-trained model. It returns the results
of the sentiment analysis as a JSON response.

The function uses the json library to parse the JSON payload and the
deepsparse package . The sentiment analysis model is defined by a
Pipeline object, which is created by specifying the task, model path,
and batch size.

Args:
event (dict):
    AWS Lambda uses this parameter to pass in event data to the handler.
    This parameter is usually of the Python dict type. It contains
    information about the API Gateway event that triggered this function.

context (object): AWS Lambda uses this parameter to provide runtime
    information to your handler. This parameter is of the LambdaContext type.

Returns:
A dictionary containing the HTTP status code and the sentiment analysis results
as a JSON response.
"""

import json

from deepsparse import Pipeline


pipeline = Pipeline.create(
    task="sentiment_analysis", model_path="./model/deployment", batch_size=1
)


def lambda_handler(event, context):

    payload = json.loads(event["body"])
    inference = pipeline(**payload)

    return {
        "statusCode": 200,
        "body": inference.json(),
    }
