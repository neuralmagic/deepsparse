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
import os
import csv

from deepsparse import Pipeline

s3_client = boto3.client('s3')

pipeline = Pipeline.create(
    task="sentiment_analysis",
    model_path="./model/deployment"
)

input_str = os.environ.get('INPUTS')
input_lst = input_str.split(',')

inference = pipeline(input_lst)

def write_list_to_csv(my_list, csv_file_name):

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_file_path = os.path.join(output_dir, csv_file_name)
    print(csv_file_path)

    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for element in my_list:
            writer.writerow([element])
            
            
def upload_file_to_s3(file_path, bucket_name, s3_key):
    
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(file_path, s3_key)


if __name__ == '__main__':
    write_list_to_csv(inference.labels, "outputs.csv" )
    upload_file_to_s3('./output/outputs.csv', 'batch-output-deepsparse', 'outputs.csv')