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

import asyncio

import yaml

from deepsparse.transformers import pipeline
from tweepy.asynchronous import AsyncStream
from usernames import user_id, user_name


def get_config(path):

    with open(path) as file:
        config = yaml.safe_load(file.read())

    return config


token_path = "./config.yaml"
model_path = get_config(token_path)
text_classification = pipeline(
    task="text-classification", model_path=model_path["model"]
)


class SparseStream(AsyncStream):
    async def on_status(self, status):

        if (
            (not status.retweeted)
            and ("RT @" not in status.text)
            and (status.in_reply_to_screen_name not in user_name)
            and (status.in_reply_to_status_id is None)
        ):

            inference = text_classification(status.text)[0]
            inference = "positive" if inference["label"] == "LABEL_1" else "negative"
            output = {"tweet": status.text, "sentiment": inference}
            print(output)


async def main(token):

    token = token["twitter_tokens"]
    stream = SparseStream(
        token["consumer_key"],
        token["consumer_secret"],
        token["access_token"],
        token["access_token_secret"],
    )

    await stream.filter(follow=user_id, stall_warnings=True)


if __name__ == "__main__":

    token = get_config(token_path)
    asyncio.run(main(token))
