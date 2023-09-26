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

import math
from typing import List

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.benchmark.benchmark_pipeline import calculate_section_stats
from deepsparse.benchmark.config import PipelineBenchmarkConfig
from deepsparse.benchmark.data_creation import (
    SchemaType,
    generate_random_image_data,
    generate_random_question_data,
    generate_random_text_data,
    get_input_schema_type,
)
from deepsparse.utils import StagedTimer, TimerManager
from tests.helpers import run_command


@pytest.mark.parametrize(
    ("pipeline_id", "model_stub", "additional_opts"),
    [
        (
            "text_classification",
            "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/"
            "sst2/pruned90-none",
            [
                "-c",
                "tests/test_data/pipeline_bench_config.json",
                "-b",
                "4",
                "-t",
                "3",
                "-w",
                "1",
            ],
        ),
        (
            "image_classification",
            "zoo:cv/classification/resnet_v1-50_2x/pytorch/sparseml/imagenet/base-none",
            [
                "-c",
                "tests/test_data/pipeline_bench_config.json",
                "-s",
                "async",
                "-t",
                "3",
                "-w",
                "1",
            ],
        ),
        (
            "image_classification",
            "zoo:cv/classification/resnet_v1-50_2x/pytorch/sparseml/imagenet/base-none",
            [],
        ),
        (
            "token_classification",
            "zoo:nlp/token_classification/distilbert-none/pytorch/huggingface/"
            "conll2003/pruned90-none",
            [
                "-c",
                "tests/test_data/pipeline_bench_config.json",
                "-s",
                "elastic",
                "-t",
                "3",
                "-w",
                "1",
            ],
        ),
    ],
)
def test_pipeline_benchmark(
    pipeline_id: str, model_stub: str, additional_opts: List[str]
):
    cmd = [
        "deepsparse.benchmark_pipeline",
        pipeline_id,
        model_stub,
        *additional_opts,
    ]
    print(f"\n==== test_benchmark command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_benchmark output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()
    assert "total_inference" in res.stdout.lower()


def test_generate_random_image_data():
    batch_size = 32
    config_args = {"input_image_shape": (600, 600, 1)}
    config = PipelineBenchmarkConfig(**config_args)
    image_data = generate_random_image_data(config, batch_size)
    assert len(image_data) == batch_size
    img = image_data[0]
    assert img.shape == (600, 600, 1)
    assert img.dtype == numpy.uint8
    assert numpy.max(img) < 255 and numpy.min(img) >= 0


def test_generate_random_text_data():
    batch_size = 16
    avg_word_len = 8
    config_args = {"gen_sequence_length": 250}
    config = PipelineBenchmarkConfig(**config_args)
    text_data = generate_random_text_data(config, batch_size, avg_word_len=avg_word_len)
    assert len(text_data) == batch_size
    text = text_data[0]
    assert len(text) == 250
    num_spaces = text.count(" ")
    assert num_spaces == int(len(text) / avg_word_len)


def test_generate_random_question_data():
    avg_word_len = 10
    config_args = {"gen_sequence_length": 50}
    config = PipelineBenchmarkConfig(**config_args)
    question, context = generate_random_question_data(
        config, 1, avg_word_len=avg_word_len
    )
    assert len(question) == config.gen_sequence_length
    assert len(context) == config.gen_sequence_length
    num_q_spaces = question.count(" ")
    num_c_spaces = context.count(" ")
    assert num_q_spaces == num_c_spaces == int(len(question) / avg_word_len)


@pytest.mark.parametrize(
    ("task_name", "input_schema"),
    [
        ("yolo", SchemaType.IMAGE),
        ("text_classification", SchemaType.TEXT_SEQ),
        ("transformers_embedding_extraction", SchemaType.TEXT_INPUT),
        ("question_answering", SchemaType.QUESTION),
    ],
)
def test_get_input_schema_type(task_name, input_schema):
    pipeline = Pipeline.create(task=task_name)
    assert get_input_schema_type(pipeline) == input_schema


def test_calculations():
    batch_times = []
    timer_manager = TimerManager()
    for i in range(5):
        timer = StagedTimer()
        timer._staged_start_times["stage_1"] = [i + 0.1]
        timer._staged_stop_times["stage_1"] = [i + 0.5]

        timer._staged_start_times["stage_2"] = [i + 0.6]
        timer._staged_stop_times["stage_2"] = [i + 0.9]

        timer_manager._timers.append(timer)

    batch_times = timer_manager.all_times
    total_run_time = 6.0
    section_stats = calculate_section_stats(batch_times, total_run_time, 1)
    assert math.isclose(
        section_stats["stage_1"]["total_percentage"], 33.33, rel_tol=0.05
    )
    assert math.isclose(section_stats["stage_2"]["total_percentage"], 25, rel_tol=0.05)
    assert math.isclose(section_stats["stage_1"]["mean"], 400, rel_tol=0.05)
    assert math.isclose(section_stats["stage_2"]["median"], 300, rel_tol=0.05)
