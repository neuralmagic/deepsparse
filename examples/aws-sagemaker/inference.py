import boto3
import json

class Endpoint:

    def __init__(self, region_name, endpoint_name):

        self.region_name = region_name
        self.endpoint_name = endpoint_name
        self.content_type = "application/json"
        self.accept = "text/plain"
        self.client = boto3.client(
            "sagemaker-runtime", region_name=self.region_name
        )

    def predict(self, question, context):

        body = json.dumps(
            dict(
                question=question,
                context=context,
            )
        )
        res = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=body,
            ContentType=self.content_type,
            Accept=self.accept,
        )

        print(res["Body"].readlines())