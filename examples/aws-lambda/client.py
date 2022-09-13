import json

import requests


class LambdaClient:
    """
    Client object for making requests to the Lambda HTTP endpoint
    :param url: API endpoint URL
    """

    def __init__(self, url: str):

        self.url = url
        self.headers = {'Content-Type': 'application/json'}

    def qa_client(self, question: str, context: str) -> bytes:

        """
        :param question: question input to the model pipeline.
        :param context: context input to the model pipeline.
        :return: json output from Lambda
        """

        obj = {
            'question': question,
            'context': context
        }

        response = requests.post(self.url, headers=self.headers, json=obj)

        return json.loads(response.content)
