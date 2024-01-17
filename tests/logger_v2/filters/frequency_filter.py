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
from deepsparse.loggers_v2.filters import FrequencyFilter


@pytest.mark.parametrize(
    "tag,func,rate, expected_counter",  # From config file
    [
        ("(?i)operator", "max", 2, 311 // 2),
        ("(?i)operator", "max", 3, 311 // 3),
    ],
)
def test_frequency_filter(tag, func, rate, expected_counter):
    """basic filtering test by frequency"""
    freq_filter = FrequencyFilter()
    freq_filter.add_template_to_frequency(tag, func, rate)

    counter = 0
    for _ in range(311):
        if freq_filter.should_execute_on_frequency(
            tag="operator",
            log_type="system",
            func=func,
        ):
            counter += 1

    assert counter == expected_counter


@pytest.mark.parametrize(
    "freq, iter, expected_times_to_log",  # From config file
    [
        (
            1,
            1,
            {
                "AutoRegressiveOperatorPreprocess": 1,
                "NLEngineOperator": 1,
                "GenerateNewTokenOperator": 1,
            },
        ),
        (
            2,
            1,
            {
                "AutoRegressiveOperatorPreprocess": 1,
                "NLEngineOperator": 1,
                "GenerateNewTokenOperator": 1,
            },
        ),
        (
            3,
            1,
            {
                "AutoRegressiveOperatorPreprocess": 0,
                "NLEngineOperator": 1,
                "GenerateNewTokenOperator": 1,
            },
        ),
        (
            4,
            1,
            {
                "AutoRegressiveOperatorPreprocess": 0,
                "NLEngineOperator": 0,
                "GenerateNewTokenOperator": 0,
            },
        ),
        (
            4,
            2,
            {
                "AutoRegressiveOperatorPreprocess": 1,
                "NLEngineOperator": 1,
                "GenerateNewTokenOperator": 1,
            },
        ),
        (
            4,
            12,
            {
                "AutoRegressiveOperatorPreprocess": 6,
                "NLEngineOperator": 9,
                "GenerateNewTokenOperator": 9,
            },
        ),
    ],
)
def test_frequency_filter_with_multiple_tags(freq, iter, expected_times_to_log):
    """Test if the given tag should be logged

    AutoRegressiveOperatorPreprocess matches with Auto, Operator, max_freq for this to log is 2
    NLEngineOperator matches with Engine, (Token|Engine), Operator, max_freq for this to log is 3
    GenerateNewTokenOperator matches with Token, (Token|Engine), Operator, max_freq for this to log is 3

    f(log) = floor(matches x iter / freq) for each func, for each root logger
    """

    tags_from_config = [
        "Auto",  # only one match
        "Engine",  # only one match
        "Token",  # only one match
        "(Token|Engine)",  # two matches
        "Operator",  # three matches
    ]

    freq_filter = FrequencyFilter()

    for tag in tags_from_config:
        freq_filter.add_template_to_frequency(tag, "func", freq)

    tags_to_log = {key: 0 for key in expected_times_to_log.keys()}
    for _ in range(iter):
        for tag_to_log in tags_to_log.keys():
            if freq_filter.should_execute_on_frequency(
                tag=tag_to_log,
                log_type="system",
                func="func",
            ):
                tags_to_log[tag_to_log] += 1

    assert tags_to_log == expected_times_to_log


@pytest.mark.parametrize(
    "freq, iter, funcs, expected_times_to_log",  # From config file
    [
        (
            1,
            1,
            ["foo", "bar"],
            {
                "AutoRegressiveOperatorPreprocess": 1,
                "NLEngineOperator": 1,
                "GenerateNewTokenOperator": 1,
            },
        ),
        (
            1,
            1,
            ["foo", "foo"],
            {
                "AutoRegressiveOperatorPreprocess": 2,
                "NLEngineOperator": 2,
                "GenerateNewTokenOperator": 2,
            },
        ),
        (
            1,
            1,
            ["bar", "bar"],
            {
                "AutoRegressiveOperatorPreprocess": 0,
                "NLEngineOperator": 0,
                "GenerateNewTokenOperator": 0,
            },
        ),
    ],
)
def test_frequency_filter_with_multiple_func(freq, iter, funcs, expected_times_to_log):
    """Test if the logs are separated by func

    AutoRegressiveOperatorPreprocess matches with Auto, Operator, max_freq for this to log is 2
    NLEngineOperator matches with Engine, (Token|Engine), Operator, max_freq for this to log is 3
    GenerateNewTokenOperator matches with Token, (Token|Engine), Operator, max_freq for this to log is 3

    f(log) = floor(matches x iter / freq) for each func, for each root logger
    """

    tags_from_config = [
        "Auto",  # only one match
        "Engine",  # only one match
        "Token",  # only one match
        "(Token|Engine)",  # two matches
        "Operator",  # three matches
    ]

    freq_filter = FrequencyFilter()

    for tag in tags_from_config:
        freq_filter.add_template_to_frequency(tag, "foo", freq)

    tags_to_log = {key: 0 for key in expected_times_to_log.keys()}
    for _ in range(iter):
        for tag_to_log in tags_to_log.keys():
            for func in funcs:
                if freq_filter.should_execute_on_frequency(
                    tag=tag_to_log,
                    log_type="system",
                    func=func,
                ):
                    tags_to_log[tag_to_log] += 1
    assert tags_to_log == expected_times_to_log


def test_frequency_filter_with_multiple_log_type():
    """Test if logs are separated by log_type
    Call using metric, one before the log satisfies freq, call using a different log_type.

    AutoRegressiveOperatorPreprocess matches with Auto, Operator, max_freq for this to log is 2
    NLEngineOperator matches with Engine, (Token|Engine), Operator, max_freq for this to log is 3
    GenerateNewTokenOperator matches with Token, (Token|Engine), Operator, max_freq for this to log is 3

    f(log) = floor(matches x iter / freq) for each func, for each root logger
    """

    freq_filter = FrequencyFilter()

    freq_filter.add_template_to_frequency("tag", "func", 2)

    counter = 0

    freq_filter.should_execute_on_frequency(
        tag="tag",
        log_type="system",
        func="func",
    )

    if freq_filter.should_execute_on_frequency(
        tag="tag",
        log_type="metric",
        func="func",
    ):
        counter += 1  # should not be here
    assert counter == 0

    if freq_filter.should_execute_on_frequency(
        tag="tag",
        log_type="system",
        func="func",
    ):
        counter += 1  # this should run

    assert counter == 1
