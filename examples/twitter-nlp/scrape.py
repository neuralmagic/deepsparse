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
Twitter scraping script using Twint.
Give a topic, or multiple, and it will pull down the desired number of tweets
that match.
Writes the results as JSON to the given output_file.
If None given, will write the results under a new file named after the given topic.


##########
Command help:
Usage: scrape.py [OPTIONS]

  Twitter scraping script using Twint. Give a topic, or multiple, and it will
  pull down the desired number of tweets that match. Writes the results as
  JSON to the given output_file. If None given, will write the results under a
  new file named after the given topic.

Options:
  -t, --topic TEXT        The topics to scrape twitter for, either keywords or
                          hashtags.For example: '--topic #crypto'. Multiple
                          topics can be used as well, for example: '-t #crypto
                          -t #bitcoin'
  --total_tweets INTEGER  The total number of tweets to gather from Twitter.
                          Note, the API used from Twitter has a maximum date
                          range of around 1 week.
  --output_file TEXT      The output file to write the tweets to. If not
                          supplied, will create a new file using the topics as
                          names.
  --help                  Show this message and exit.

##########
Example command for scraping Twitter for #crypto tweets:
python scrape.py --topic '#crypto' --total_tweets 1000
"""

from typing import List, Optional

import click
import twint


@click.command()
@click.option(
    "--topic",
    "-t",
    multiple=True,
    help="The topics to scrape twitter for, either keywords or hashtags."
    "For example: '--topic #crypto'. "
    "Multiple topics can be used as well, for example: '-t #crypto -t #bitcoin'",
)
@click.option(
    "--total_tweets",
    type=int,
    default=100,
    help="The total number of tweets to gather from Twitter. "
    "Note, the API used from Twitter has a maximum date range of around 1 week.",
)
@click.option(
    "--output_file",
    type=str,
    default=None,
    help="The output file to write the tweets to. "
    "If not supplied, will create a new file using the topics as names.",
)
def scrape_tweets(topic: List[str], total_tweets: int, output_file: Optional[str]):
    """
    Twitter scraping script using Twint.
    Give a topic, or multiple, and it will pull down the desired number of tweets
    that match.
    Writes the results as JSON lines as text to the given output_file.
    If None given, will write the results under a new file named after the given topic.
    """
    config = twint.Config()
    topics_str = " ".join(
        [f"({top})" if top.startswith("#") else top for top in topic]
    )  # reformat as hashtags
    config.Custom_query = (
        f"{topics_str} min_faves:2 lang:en -filter:links -filter:replies "
    )
    config.Limit = total_tweets
    config.Store_json = True
    config.Output = f"{'_'.join(topic)}.txt" if not output_file else output_file

    print(f"Scraping {total_tweets} tweets")
    twint.run.Search(config)
    print(f"Finished scraping, tweets written to {config.Output}")


if __name__ == "__main__":
    scrape_tweets()
