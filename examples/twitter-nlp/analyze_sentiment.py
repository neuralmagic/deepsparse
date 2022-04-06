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
Script to analyze the sentiment of a given file of tweets from Twitter
in batch processing mode.

##########
Command help:
Usage: analyze_sentiment.py [OPTIONS]

  Analyze the sentiment of the tweets given in the tweets_file and print out
  the results.

Options:
  --model_path TEXT       The path to the sentiment analysis model to
                          load.Either a model.onnx file, a model folder
                          containing the model.onnx and supporting files, or a
                          SparseZoo model stub.
  --tweets_file TEXT      The path to the tweets json txt file to analyze
                          sentiment for.
  --batch_size INTEGER    The batch size to process the tweets with. A higher
                          batch size may increase performance at the expense
                          of memory resources and individual latency.
  --total_tweets INTEGER  The total number of tweets to analyze from the
                          tweets_file.Defaults to None which will run through
                          all tweets contained in the file.
  --help                  Show this message and exit.

##########
Example running a dense, unoptimized sentiment analysis model:
python analyze_sentiment.py
    --model_path "zoo:nlp/text_classification/bert-base/pytorch/huggingface/sst2/base-none"
    --tweets_file /PATH/TO/OUTPUT/FROM/scrape.py
"""

import json
from itertools import cycle, islice
from typing import Dict, List, Optional

import click
from colorama import Fore

try:
    from deepsparse.transformers import pipeline
except:
    pass


def _load_tweets(tweets_file: str):
    tweets = []
    with open(tweets_file, "r") as file:
        for line in file.readlines():
            tweets.append(json.loads(line))

    return tweets


def _prep_data(tweets: List[Dict], total_num: int) -> List[str]:
    if total_num:
        tweets = islice(cycle(tweets), total_num)

    return [tweet["tweet"].strip().replace("\n", "") for tweet in tweets]


def _batched_model_input(tweets: List[str], batch_size: int) -> Optional[List[str]]:
    if batch_size >= len(tweets):
        return None

    batched = tweets[0:batch_size]
    del tweets[0:batch_size]

    return batched


def _display_results(batch, sentiments):
    for text, sentiment in zip(batch, sentiments):
        negative = sentiment["label"] == "LABEL_1"
        print(f"{Fore.RED if negative else Fore.CYAN}{text}")


@click.command()
@click.option(
    "--model_path",
    type=str,
    help="The path to the sentiment analysis model to load."
    "Either a model.onnx file, a model folder containing the model.onnx "
    "and supporting files, or a SparseZoo model stub.",
)
@click.option(
    "--tweets_file",
    type=str,
    help="The path to the tweets json txt file to analyze sentiment for."
)
@click.option(
    "--batch_size",
    type=int,
    default=16,
    help="The batch size to process the tweets with. "
    "A higher batch size may increase performance at the expense of memory resources "
    "and individual latency."
)
@click.option(
    "--total_tweets",
    type=int,
    default=None,
    help="The total number of tweets to analyze from the tweets_file."
    "Defaults to None which will run through all tweets contained in the file."
)
def analyze_tweets_sentiment(
    model_path: str, tweets_file: str, batch_size: int, total_tweets: int
):
    """
    Analyze the sentiment of the tweets given in the tweets_file and
    print out the results.
    """
    text_pipeline = pipeline(
        task="text-classification",
        model_path=model_path,
        batch_size=batch_size,
    )
    tweets = _load_tweets(tweets_file)
    tweets = _prep_data(tweets, total_tweets)
    tot_sentiments = []

    while True:
        batch = _batched_model_input(tweets, batch_size)
        if batch is None:
            break
        sentiments = text_pipeline(batch)
        _display_results(batch, sentiments)
        tot_sentiments.extend(sentiments)

    num_positive = sum(
        [1 if sent["label"] == "LABEL_1" else 0 for sent in tot_sentiments]
    )
    num_negative = sum(
        [1 if sent["label"] == "LABEL_0" else 0 for sent in tot_sentiments]
    )
    print("\n\n\n")
    print("###########################################################################")
    print(f"Completed analyzing {len(tweets)} tweets for sentiment,")

    if num_positive >= num_negative:
        print(
            f"{Fore.CYAN}General sentiment is positive with "
            f"{100*num_positive/float(len(tot_sentiments)):.0f}% in favor."
        )
    else:

        print(
            f"{Fore.RED}General sentiment is negative with "
            f"{100*num_negative/float(len(tot_sentiments)):.0f}% against."
        )
    print("###########################################################################")


if __name__ == "__main__":
    analyze_tweets_sentiment()
