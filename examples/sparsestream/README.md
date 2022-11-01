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


               ____  ___   ____  ____  ____  ____  ____  ___  ____  ____  ____  _  _ 
               [__   |__]  |__|  |__/  [__   |___  [__    |   |__/  |___  |__|  |\/| 
               ___]  |     |  |  |  \  ___]  |___  ___]   |   |  \  |___  |  |  |  | 
                                                                                          
      *** SparseStream classifies the sentiment and topic of finance-related tweets in a real-time stream. ***


## `INTRO`

The purpose of this app is for you to familiarize yourself with the high performance inference speeds of sparse models in a streaming environment. You will be able to use this app to classify tweets in a real-time stream.

<br />

[Getting Started with the DeepSparse Engine](https://github.com/neuralmagic/deepsparse)

## `INSTALL`

Supports Python >= 3.7

```bash
git clone https://github.com/neuralmagic/deepsparse.git
cd deepsparse/examples/sparsestream
pip install -r requirements.txt
```

## `CONFIGURING API`

In order to connect with the Twitter streaming API, access to the `consumer key`, `consumer secret`, `access token` and `access token secret` is required before getting started. These tokens will be available after applying for a Twitter app via the [Twitter developer site](https://developer.twitter.com/en). Afterwards, add these four tokens to the `config.yaml` file before starting the stream.

## `START STREAM`

In a terminal, start a Twitter stream with:
```bash
python stream.py
```

This will download sparse-optimized models and initialize two NLP inference pipelines for the financial text classification task: one pipeline for sentiment classification and the other for topic classification.

## `STREAM WHILE YOU EAT SOME POPCORN 🍿`

Tweets should now be streaming in your terminal, and you should see three objects per tweet:

- `tweet`: The tweet received from the Twitter API.
- `sentiment`: The tweet's sentiment: `Bullish` or `Bearish` or `Neutral`
- `topic`: The tweet's financial topic: Please refer to `labels.py` for a full list of the 20 topic labels.

For example:

```text
'Gold is up 25% in after-hours trading.'
'Bullish'
'Gold | Metals | Materials'
'Apple will introduce its new Ipad in November.'
'Neutral'
'Company | Product News'
'Goldman downgrades Twitter before Elon buyout.'
'Bearish'
'Analyst Update'
```
