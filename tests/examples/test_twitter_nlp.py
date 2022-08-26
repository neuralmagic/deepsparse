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

import pytest
from tests.helpers import run_command


@pytest.mark.smoke
def test_analyze_tokens(model: str, batch_size: int):
    cmd = [
        "python3",
        "examples/twitter-nlp/analyze_tokens.py",
        "--model_path",
        "zoo:nlp/token_classification/distilbert-none/pytorch"
        "/huggingface/conll2003/pruned80_quant-none-vnni",
        "--batch_size",
        "8",
        "--tweets_file",
        "tests/test_data/pineapple.txt",
    ]
    print(f"\n==== test_analyze_tokens example ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_analyze_tokens output ====\n{res.stdout}")

    # validate command executed successfully
    assert res.returncode == 0
    assert "Completed analyzing" in res.stdout.lower()


@pytest.mark.smoke
def test_analyze_sentiment(model: str, batch_size: int):
    cmd = [
        "python3",
        "examples/twitter-nlp/analyze_sentiment.py",
        "--model_path",
        "zoo:nlp/sentiment_analysis/distilbert-none/pytorch"
        "/huggingface/sst2/pruned80_quant-none-vnni",
        "--batch_size",
        "8",
        "--tweets_file",
        "tests/test_data/pineapple.txt",
    ]
    print(f"\n==== test_analyze_sentiment example ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_analyze_sentiment output ====\n{res.stdout}")

    # validate command executed successfully
    assert res.returncode == 0
    assert "Completed analyzing" in res.stdout.lower()
