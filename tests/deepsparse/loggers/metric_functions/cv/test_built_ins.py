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

import numpy

import pytest
from deepsparse.loggers.metric_functions import (
    count_classes_detected,
    count_number_objects_detected,
    fraction_zeros,
    image_shape,
    mean_pixels_per_channel,
    mean_score_per_detection,
    std_pixels_per_channel,
    std_score_per_detection,
)


def _generate_array_and_fill_with_n_zeros(fill_value, shape, n_zeros):
    array = numpy.full(fill_value=fill_value, shape=shape)
    array = array.flatten()
    array[:n_zeros] = 0.0
    array = numpy.reshape(array, shape)
    return array


BBOX = [500.0, 500.0, 400.0, 400.0]


@pytest.mark.parametrize(
    "image, expected_shape",
    [
        (numpy.random.rand(2, 3, 16, 16), {"channels": 3, "dim_0": 16, "dim_1": 16}),
        (numpy.random.rand(2, 16, 16, 3), {"channels": 3, "dim_0": 16, "dim_1": 16}),
        (numpy.random.rand(16, 16, 1), {"channels": 1, "dim_0": 16, "dim_1": 16}),
        (numpy.random.rand(16, 1, 16, 15), {"channels": 1, "dim_0": 16, "dim_1": 15}),
    ],
)
def test_image_shape(image, expected_shape):
    assert expected_shape == image_shape(image)


@pytest.mark.parametrize(
    "image, expected_means",
    [
        (
            numpy.full(fill_value=0.3, shape=(16, 3, 16, 16)),
            numpy.full(fill_value=0.3, shape=3),
        ),
        (
            numpy.full(fill_value=0.3, shape=(2, 1, 16, 16)),
            numpy.full(fill_value=0.3, shape=1),
        ),
        (
            numpy.full(fill_value=0.5, shape=(1, 16, 16)),
            numpy.full(fill_value=0.5, shape=1),
        ),
        (
            numpy.full(fill_value=0.5, shape=(3, 16, 16)),
            numpy.full(fill_value=0.5, shape=3),
        ),
        (
            numpy.full(fill_value=0.5, shape=(16, 16, 1)),
            numpy.full(fill_value=0.5, shape=1),
        ),
        (
            numpy.full(fill_value=0.5, shape=(16, 16, 3)),
            numpy.full(fill_value=0.5, shape=3),
        ),
        (
            numpy.full(fill_value=0.5, shape=(16, 16, 15, 3)),
            numpy.full(fill_value=0.5, shape=3),
        ),
        (
            numpy.full(fill_value=0.5, shape=(16, 15, 16, 3)),
            numpy.full(fill_value=0.5, shape=3),
        ),
    ],
)
def test_mean_pixel_per_channel(image, expected_means):
    result = mean_pixels_per_channel(image)
    for mean, expected_mean in zip(result.values(), expected_means):
        numpy.testing.assert_allclose(mean, expected_mean)
    assert list(result.keys()) == [
        f"mean_channel_{idx}" for idx in range(len(expected_means))
    ]


@pytest.mark.parametrize(
    "image, expected_stds",
    [
        (numpy.full(fill_value=0.3, shape=(2, 3, 16, 16)), numpy.zeros(3)),
        (numpy.full(fill_value=0.3, shape=(2, 1, 16, 16)), numpy.zeros(1)),
        (numpy.full(fill_value=0.5, shape=(1, 16, 16)), numpy.zeros(1)),
        (numpy.full(fill_value=0.5, shape=(16, 16, 1)), numpy.zeros(1)),
        (numpy.full(fill_value=0.5, shape=(3, 16, 16)), numpy.zeros(3)),
    ],
)
def test_std_pixels_per_channel(image, expected_stds):
    result = std_pixels_per_channel(image)
    for std, expected_std in zip(result.values(), expected_stds):
        numpy.testing.assert_allclose(std, expected_std, atol=1e-16)
    assert list(result.keys()) == [
        f"std_channel_{idx}" for idx in range(len(expected_stds))
    ]


@pytest.mark.parametrize(
    "image, expected_percentage",
    [
        (
            _generate_array_and_fill_with_n_zeros(
                fill_value=0.5, shape=(2, 16, 16, 3), n_zeros=0
            ),
            0.0,
        ),
        (
            _generate_array_and_fill_with_n_zeros(
                fill_value=120, shape=(3, 3, 16, 16), n_zeros=3 * 3 * 16 * 8
            ),
            0.5,
        ),
        (
            _generate_array_and_fill_with_n_zeros(
                fill_value=0.1, shape=(16, 16, 1), n_zeros=16 * 16 * 1
            ),
            1.0,
        ),
    ],
)
def test_percentage_zeros_per_channel(image, expected_percentage):
    assert expected_percentage == fraction_zeros(image)


@pytest.mark.parametrize(
    "classes, expected_count_classes, should_raise_error",
    [
        ([[None], [0, 1, 3], [3, 3, 0]], {"0": 2, "1": 1, "3": 3}, False),
        ([[None], [None]], {}, False),
        (
            [["foo", "bar"], ["foo", "bar", "alice"]],
            {"foo": 2, "bar": 2, "alice": 1},
            False,
        ),
        (
            [["foo", "bar"], ["foo", "bar", "alice"], [None]],
            {"foo": 2, "bar": 2, "alice": 1},
            False,
        ),
        ([[6.666], [0, 1, 3], [3, 3, 0]], None, True),
        ([[None, None], [0, 1, 3], [3, 3, 0]], None, True),
    ],
)
def test_count_classes_detected(classes, expected_count_classes, should_raise_error):
    if should_raise_error:
        with pytest.raises(ValueError):
            count_classes_detected(classes)
        return
    assert expected_count_classes == count_classes_detected(classes)


@pytest.mark.parametrize(
    "classes, expected_count_classes",
    [
        ([[None], [0, 1, 3], [3, 3, 0]], {"0": 0, "1": 3, "2": 3}),
        ([[None], [None]], {"0": 0, "1": 0}),
        ([["foo", "bar"], ["foo", "bar", "alice"], ["bar"]], {"0": 2, "1": 3, "2": 1}),
        ([["foo", "bar"], ["foo", "bar", "alice"], [None]], {"0": 2, "1": 3, "2": 0}),
    ],
)
def test_count_number_objects_detected(classes, expected_count_classes):
    assert expected_count_classes == count_number_objects_detected(classes)


@pytest.mark.parametrize(
    "scores, expected_mean_score",
    [
        ([[None], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]], {"0": 0.0, "1": 0.5, "2": 0.3}),
        ([[None], [None]], {"0": 0.0, "1": 0.0}),
        ([[0.5, 0.5], [0.9, 0.9, 0.9], [1.0]], {"0": 0.5, "1": 0.9, "2": 1.0}),
        ([[1.0, 0.0], [None]], {"0": 0.5, "1": 0.0}),
    ],
)
def test_mean_score_per_detection(scores, expected_mean_score):
    assert expected_mean_score == mean_score_per_detection(scores)


@pytest.mark.parametrize(
    "scores, expected_std_score",
    [
        (
            [[None], [0.5, 0.5, 0.5], [0.6, 0.7, 0.8]],
            {"0": 0.0, "1": 0.0, "2": 0.08164},
        ),
        ([[None], [None]], {"0": 0.0, "1": 0.0}),
        ([[1.0, 0.0], [None]], {"0": 0.5, "1": 0.0}),
    ],
)
def test_std_score_per_detection(scores, expected_std_score):
    result = std_score_per_detection(scores)
    for (result_keys, results_values), (keys, values) in zip(
        result.items(), expected_std_score.items()
    ):
        numpy.testing.assert_allclose(results_values, values, atol=1e-5)
        assert result_keys == keys
