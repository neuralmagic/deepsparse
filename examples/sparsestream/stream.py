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

from deepsparse import Pipeline
from labels import sentiments, topics
from tweepy.asynchronous import AsyncStream
from usernames import user_id, user_name


config_path = "./config.yaml"


def get_config(path):

    with open(path) as file:
        config = yaml.safe_load(file.read())

    return config


config = get_config(config_path)

sentiment_classifier = Pipeline.create(
    task=config["task"], model_path=config["sent_model"], scheduler="sync"
)

topic_classifier = Pipeline.create(
    task=config["task"], model_path=config["topic_model"], scheduler="sync"
)


class SparseStream(AsyncStream):
    async def on_status(self, status):

        """logic to prevent retweets and replies to appearing in stream"""

        if (
            (not status.retweeted)
            and ("RT @" not in status.text)
            and (status.in_reply_to_screen_name not in user_name)
            and (status.in_reply_to_status_id is None)
        ):

            sentiment = sentiment_classifier(status.text)
            sentiment = sentiments[sentiment.labels[0]]
            topic = topic_classifier(status.text)
            topic = topics[topic.labels[0]]

            output = {"tweet": status.text, "sentiment": sentiment, "topic": topic}

            print(output)


async def main():

    stream = SparseStream(
        config["consumer_key"],
        config["consumer_secret"],
        config["access_token"],
        config["access_token_secret"],
    )

    await stream.filter(follow=user_id, stall_warnings=True)


if __name__ == "__main__":

    asyncio.run(main())
