<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Twitter NLP Inference Examples

This directory contains examples for scraping, processing, and classifying Twitter data
using the DeepSparse engine for >=10x increase in inference performance on commodity CPUs.

## Installation

The dependencies for this example can be installed using `pip`:
```bash
pip3 install -r requirements.txt
```

## Sentiment Analysis Example

The `analyze_sentiment.py` script is used to analyze and classify tweets as either positive or negative
depending on their contents. 
For example, you can analyze the general sentiment of crypto or other common topics across Twitter.

To use, first gather the desired number of tweets for your topic(s) and save them as a text file to use with `analyze_sentiment.py`.
The script expects one tweet per row with each tweet formatted as a json object containing a `"tweet"` key that maps to the text content.

An example script `scrape.py` is given to show this in action.
Note, it uses the Twint library which does not abide by Twitter's terms of service. 
The script is given as an example only and users are expected to use Twitter's developer pathways and APIs in place of this script.
```bash
python scrape.py --topic '#crypto' --total_tweets 1000
```

Next, use the `analyze_sentiment.py` along with sparsified sentiment analysis models from the [SparseZoo](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=sentiment_analysis&page=1)
to performantly analyze the general sentiment across the gathered tweets:
```bash
python analyze_sentiment.py \
    --model_path "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/12layer_pruned80_quant-none-vnni" \
    --tweets_file "#crypto.txt"
```
