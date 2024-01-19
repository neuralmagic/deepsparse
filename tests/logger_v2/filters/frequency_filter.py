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

from collections import defaultdict
from threading import Lock
from typing import Any, Callable

import pytest
from deepsparse.loggers_v2.filters import FrequencyFilter
from deepsparse.loggers_v2.filters.pattern import is_match_found


@pytest.mark.parametrize(
    "tag, func, freq, expected_counter, iter",  # From config file
    [
        ("(?i)operator", "max", 2, 6, 12),
        ("(?i)operator", "max", 3, 4, 12),
        ("(?i)operator", "max", 5, 2, 12),
    ],
)
def test_frequency_filter(tag, func, freq, expected_counter, iter):
    """basic filtering test by frequency"""
    freq_filter = FrequencyFilter()

    counter = 0
    for _ in range(iter):
        freq_filter.inc(tag, func)

        if freq_filter.should_execute_on_frequency(
            tag=tag,
            func=func,
            freq=freq,
        ):
            counter += 1
    stub = f"{tag}.{func}"
    assert counter == expected_counter
    assert freq_filter.counter[stub] == iter


@pytest.mark.parametrize(
    "tag_freq_func, iter, expected_counter_calls",  # From config file
    [
        (  # unique tag, same func
            [
                ("tag1", 1, "func"),
                ("tag2", 3, "func"),
                ("tag3", 7, "func"),
            ],
            15,
            {
                "tag1.func": 15,
                "tag2.func": 5,
                "tag3.func": 2,
            },
        ),
        (  # duplicated tag1.func
            [
                ("tag1", 1, "func"),
                ("tag1", 3, "func"),
                ("tag3", 7, "func"),
            ],
            15,
            {
                "tag1.func": 15 + 5,
                "tag3.func": 2,
            },
        ),
        (  # duplicated tag1
            [
                ("tag1", 3, "func"),
                ("tag1", 3, "func2"),
                ("tag3", 7, "func3"),
            ],
            15,
            {
                "tag1.func": 5,
                "tag1.func2": 5,
                "tag3.func3": 2,
            },
        ),
        (  # tag, func being shared
            [
                ("tag1", 3, "func"),
                ("tag1", 3, "func2"),
                ("tag3", 7, "func"),
                ("tag3", 5, "func3"),
            ],
            15,
            {
                "tag1.func": 5,
                "tag1.func2": 5,
                "tag3.func": 2,
                "tag3.func3": 3,
            },
        ),
    ],
)
def test_frequency_filter_with_tag_freq_func_combination(
    tag_freq_func, iter, expected_counter_calls
):
    """Test to check the regex number of matches with respect to the input tag"""

    freq_filter = FrequencyFilter()
    counter = defaultdict(int)

    for tag, freq, func in tag_freq_func:
        stub = f"{tag}.{func}"

        for _ in range(iter):
            freq_filter.inc(tag, func)

            if freq_filter.should_execute_on_frequency(
                tag=tag,
                func=func,
                freq=freq,
            ):
                counter[stub] += 1
    assert counter == expected_counter_calls
