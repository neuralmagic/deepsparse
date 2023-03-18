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

import boto3

s3_client = boto3.client('s3')
batch_client = boto3.client('batch')


def lambda_handler(event, context):
    """
    This lambda function is triggered by an S3 event when a file is added to the specified bucket. 
    The function reads the contents of the file, splits it into lines, and submits a batch job 
    using AWS Batch service. The batch job runs a Python script app.py with environment variables 
    set to the input data in the file.

    Args:
    event (dict): 
        AWS Lambda uses this parameter to pass in event data to the handler. 
        This parameter is usually of the Python dict type. It contains information about 
        the S3 object added 
    context (object): 
        AWS Lambda uses this parameter to provide runtime information to your handler. 
        This parameter is of the LambdaContext type.

    """
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    response = s3_client.get_object(Bucket=bucket, Key=key)
    
    strings = response['Body'].read().decode('utf-8')
    strings = strings.split('\n')
    
    string_data = []
    for row in strings:
        if row:
            string_data.append(row)
    
    response = batch_client.submit_job(
        jobName='batch-job',
        jobQueue='deepsparse-batch',
        jobDefinition='batch-definition',
        containerOverrides={
            'command': ['python', 'app.py'],
            'environment': [
                {'name': 'INPUTS', 'value': ','.join(string_data)}
            ]
        }
    )

    return {
        "statusCode": 200,
        "body": response,
    }