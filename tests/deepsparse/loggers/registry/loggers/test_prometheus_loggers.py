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

import requests

import pytest
from deepsparse.legacy.loggers import MetricCategories
from deepsparse.loggers.filters import unravel_value_as_generator
from deepsparse.loggers.registry.loggers.prometheus_logger import (
    PrometheusLogger,
    get_prometheus_metric,
)
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Summary
from tests.helpers import find_free_port
from tests.utils import mock_engine


@pytest.mark.parametrize(
    "tag, log_type, registry, expected_metric",
    [
        ("dummy_tag", "metric", REGISTRY, Summary),
        ("dummy_tag", "system", REGISTRY, None),
        (
            "prediction_latency/dummy_tag",
            "system",
            REGISTRY,
            Histogram,
        ),
        (
            "resource_utilization/dummy_tag",
            "system",
            REGISTRY,
            Gauge,
        ),
        (
            "request_details/successful_request",
            "system",
            REGISTRY,
            Counter,
        ),
        (
            "request_details/input_batch_size",
            "system",
            REGISTRY,
            Histogram,
        ),
        (
            "request_details/response_message",
            "system",
            REGISTRY,
            None,
        ),
    ],
)
def test_get_prometheus_metric(tag, log_type, registry, expected_metric):
    metric = get_prometheus_metric(tag, log_type, registry)

    if metric is None:
        assert metric is expected_metric
        return
    assert isinstance(metric, expected_metric)
    assert (
        metric._documentation
        == "{metric_type} metric for tag: {tag} | log_type: {log_type}".format(  # noqa: E501
            metric_type=metric._type, tag=tag, log_type=log_type
        )
    )


@pytest.mark.parametrize(
    "tag, no_iterations, value,",
    [
        ("dummy_pipeline/dummy.tag_1", 2, 999.0),
        ("dummy_pipeline/dummy.tag_2", 20, 1234),
    ],
)
@mock_engine(rng_seed=0)
def test_prometheus_logger(
    engine,
    tmp_path,
    tag,
    no_iterations,
    value,
):
    port = find_free_port()
    logger = PrometheusLogger(
        port=port,
        text_log_save_dir=tmp_path,
    )

    for _ in range(no_iterations):
        logger.log(tag, value, "metric")

    response = requests.get(f"http://0.0.0.0:{port}").text
    request_log_lines = response.split("\n")

    # index -5 is where we get '{tag}_count {no_iterations}'
    count_request_request = float(request_log_lines[-6].split(" ")[1])

    with open(logger.text_log_file_path) as f:
        text_log_lines = f.readlines()
    count_request_text = float(text_log_lines[-5].split(" ")[1])

    assert count_request_request == count_request_text == no_iterations


@pytest.mark.parametrize(
    "tag, value, expected_logs",
    [
        (
            "dummy_tag",
            {"foo": {"alice": 1, "bob": [1, 2, 3]}, "bar": 5},
            {
                'deepsparse_dummy_tag__foo__alice___count{pipeline_name="dummy_tag"} 1.0',  # noqa: E501
                'deepsparse_dummy_tag__foo__bob__2___count{pipeline_name="dummy_tag"} 1.0',  # noqa: E501
                'deepsparse_dummy_tag__foo__bob__2___sum{pipeline_name="dummy_tag"} 3.0',  # noqa: E501
                'deepsparse_dummy_tag__bar___sum{pipeline_name="dummy_tag"} 5.0',  # noqa: E501
            },
        ),
    ],
)
@mock_engine(rng_seed=0)
def test_nested_value_inputs(engine, tag, value, expected_logs):
    port = find_free_port()
    logger = PrometheusLogger(port=port)
    for capture, val in unravel_value_as_generator(value, tag):
        logger.log(tag=tag, value=val, log_type="metric", capture=capture)

    response = requests.get(f"http://0.0.0.0:{port}").text
    request_log_lines = response.split("\n")
    assert set(request_log_lines).issuperset(expected_logs)


@pytest.mark.parametrize(
    "tag, additional_args, expected_logs",
    [
        (
            "some_dummy_tag",
            {"pipeline_name": "dummy_pipeline"},
            {
                'deepsparse_some_dummy_tag_count{pipeline_name="some_dummy_tag"} 1.0',  # noqa: E501
                'deepsparse_some_dummy_tag_sum{pipeline_name="some_dummy_tag"} 1.0',  # noqa: E501
            },
        ),
    ],
)
@mock_engine(rng_seed=0)
def test_using_labels(engine, tag, additional_args, expected_logs):
    port = find_free_port()
    logger = PrometheusLogger(port=port)
    logger.log(
        tag=tag,
        value=1.0,
        log_type=MetricCategories.DATA,
        **additional_args,
    )
    response = requests.get(f"http://0.0.0.0:{port}").text
    request_log_lines = response.split("\n")
    assert set(request_log_lines).issuperset(expected_logs)
