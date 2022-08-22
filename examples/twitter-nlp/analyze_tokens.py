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

# flake8: noqa

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
Example running a sparse, quantized sentiment analysis model:
python analyze_sentiment.py
    --model_path "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/12layer_pruned80_quant-none-vnni"
    --tweets_file /PATH/TO/OUTPUT/FROM/scrape.py

##########
Example running a dense, unoptimized sentiment analysis model:
python analyze_sentiment.py
    --model_path "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/base-none"
    --tweets_file /PATH/TO/OUTPUT/FROM/scrape.py
"""

import json
import time
from itertools import cycle, islice
from typing import Dict, List, Optional

import click

from deepsparse import Pipeline
from rich import print


ner_tag_map = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}


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
    if batch_size > len(tweets):
        return None

    batched = tweets[0:batch_size]
    del tweets[0:batch_size]

    return batched


def _extract_important_tokens(tokens: List, important_token: str):
    # loc_tokens = []
    # for i, token in enumerate(tokens):
    #     if "LOC" in token.entity:
    #         if token.word.startswith("##"):

    loc_tokens = [token for token in tokens if important_token in token.entity]

    loc_words = [token.word for token in loc_tokens]
    compressed_loc_words = []
    for word in loc_words:
        if not word.startswith("##") or not compressed_loc_words:
            compressed_loc_words.append(word)
        else:
            compressed_loc_words[-1] += word[2:]
    return loc_tokens, compressed_loc_words


def _display_results(batch, batch_pred, important_token: str):
    for text, tokens in zip(batch, batch_pred):
        loc_tokens, cr_loc_words = _extract_important_tokens(tokens, important_token)

        print()
        print(text)

        if len(cr_loc_words) > 0:
            color = "magenta"
            print(
                f"Found {len(cr_loc_words)} important tokens: [{color}]{cr_loc_words}[/{color}]"
            )
            print(loc_tokens)


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
    help="The path to the tweets json txt file to analyze sentiment for.",
)
@click.option(
    "--batch_size",
    type=int,
    default=16,
    help="The batch size to process the tweets with. "
    "A higher batch size may increase performance at the expense of memory resources "
    "and individual latency.",
)
@click.option(
    "--total_tweets",
    type=int,
    default=None,
    help="The total number of tweets to analyze from the tweets_file."
    "Defaults to None which will run through all tweets contained in the file.",
)
@click.option(
    "--engine",
    type=click.Choice(["deepsparse", "onnxruntime"]),
    default="deepsparse",
)
@click.option(
    "--important_token",
    type=click.Choice(["MIS", "PER", "ORG", "LOC"]),
    default="LOC",
    help="Which tokens to extract: "
    "'PER' for people, 'ORG' for organizations, 'LOC' for locations",
)
def analyze_tweets_sentiment(
    model_path: str,
    tweets_file: str,
    batch_size: int,
    total_tweets: int,
    engine: str,
    important_token: str,
):
    """
    Analyze the sentiment of the tweets given in the tweets_file and
    print out the results.
    """
    print("Loading the model for inference...")
    token_classify = Pipeline.create(
        task="token-classification",
        model_path=model_path,
        batch_size=batch_size,
        engine_type=engine,
    )

    tweets = _load_tweets(tweets_file)
    tweets = _prep_data(tweets, total_tweets)
    num_tweets = len(tweets)
    tot_tokens = []
    times = []

    while True:
        batch = _batched_model_input(tweets, batch_size)
        if batch is None:
            break
        start = time.time()
        tokens = token_classify(inputs=batch)
        end = time.time()

        tokens = tokens.predictions

        _display_results(batch, tokens, important_token)
        tot_tokens.extend(tokens)
        times.append(end - start)

    print("\n\n\n")
    print("###########################################################################")
    print(
        f"Completed analyzing {len(tot_tokens)} tweets for a total of {len([item for sublist in tot_tokens for item in sublist])} tokens extracted."
    )
    print(f"This took {sum(times):2f} seconds total")
    print("###########################################################################")


if __name__ == "__main__":
    analyze_tweets_sentiment()
