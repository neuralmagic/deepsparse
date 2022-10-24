import json

import requests


class CloudRunClient:
    """
    Client object for making requests to the CLoud Run HTTP endpoint
    :param url: API endpoint URL
    """

    def __init__(self, url: str):

        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def client(self, **kwargs):
        """
        Client for question answering task.
        :param question: question input to the model pipeline.
        :param context: context input to the model pipeline.
        :return: json output from Lambda
        """

        response = requests.post(self.url, headers=self.headers, json=kwargs)

        return json.loads(response.content)