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
import torch
from deepsparse.loggers.built_ins import (
    fraction_zeros,
    image_shape,
    max_pixels_per_channel,
    mean_pixels_per_channel,
    num_bounding_boxes,
    std_pixels_per_channel,
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
        (numpy.random.rand(2, 3, 16, 16), (3, 16, 16)),
        (numpy.random.rand(2, 16, 16, 3), (16, 16, 3)),
        (numpy.random.rand(16, 16, 1), (16, 16, 1)),
        (torch.rand(2, 3, 16, 16), (3, 16, 16)),
    ],
)
def test_image_shape(image, expected_shape):
    assert expected_shape == image_shape(image)


@pytest.mark.parametrize(
    "image, expected_means",
    [
        (
            numpy.full(fill_value=0.3, shape=(2, 3, 16, 16)),
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
            numpy.full(fill_value=0.5, shape=(16, 16, 1)),
            numpy.full(fill_value=0.5, shape=1),
        ),
        (
            numpy.full(fill_value=0.5, shape=(3, 16, 16)),
            numpy.full(fill_value=0.5, shape=3),
        ),
        (
            torch.full(fill_value=120, size=(16, 16, 3)),
            numpy.full(fill_value=120, shape=3),
        ),
    ],
)
def test_mean_pixel_per_channel(image, expected_means):
    numpy.testing.assert_allclose(expected_means, mean_pixels_per_channel(image))


@pytest.mark.parametrize(
    "image, expected_stds",
    [
        (numpy.full(fill_value=0.3, shape=(2, 3, 16, 16)), numpy.zeros(3)),
        (numpy.full(fill_value=0.3, shape=(2, 1, 16, 16)), numpy.zeros(1)),
        (numpy.full(fill_value=0.5, shape=(1, 16, 16)), numpy.zeros(1)),
        (numpy.full(fill_value=0.5, shape=(16, 16, 1)), numpy.zeros(1)),
        (numpy.full(fill_value=0.5, shape=(3, 16, 16)), numpy.zeros(3)),
        (torch.full(fill_value=120, size=(16, 16, 3)), numpy.zeros(3)),
    ],
)
def test_std_pixels_per_channel(image, expected_stds):
    numpy.testing.assert_allclose(
        expected_stds, std_pixels_per_channel(image), atol=1e-16
    )


@pytest.mark.parametrize(
    "image, expected_max",
    [
        (
            numpy.full(fill_value=0.3, shape=(2, 3, 16, 16)),
            numpy.full(fill_value=0.3, shape=3),
        ),
        (
            numpy.full(fill_value=0.5, shape=(3, 16, 16)),
            numpy.full(fill_value=0.5, shape=3),
        ),
        (
            numpy.full(fill_value=120, shape=(16, 16, 1)),
            numpy.full(fill_value=120, shape=1),
        ),
        (
            torch.full(fill_value=0.5, size=(3, 16, 16, 3)),
            numpy.full(fill_value=0.5, shape=3),
        ),
    ],
)
def test_max_pixels_per_channel(image, expected_max):
    numpy.testing.assert_allclose(
        expected_max, max_pixels_per_channel(image), atol=1e-16
    )


@pytest.mark.parametrize(
    "image, expected_percentage",
    [
        (_generate_array_and_fill_with_n_zeros(0.5, (2, 16, 16, 3), 0), 0.0),
        (
            _generate_array_and_fill_with_n_zeros(120, (3, 3, 16, 16), 3 * 3 * 16 * 8),
            0.5,
        ),
        (_generate_array_and_fill_with_n_zeros(0.1, (16, 16, 1), 16 * 16 * 1), 1.0),
    ],
)
def test_percentage_zeros_per_channel(image, expected_percentage):
    assert expected_percentage == fraction_zeros(image)


@pytest.mark.parametrize(
    "bboxes, expected_num_bboxes",
    [
        ([BBOX, BBOX, BBOX, BBOX, BBOX], 5),
        ([[BBOX, BBOX, BBOX]], 3),
        ([[]], 0),
        ([], 0),
    ],
)
def test_num_bounding_boxes(bboxes, expected_num_bboxes):
    assert expected_num_bboxes == num_bounding_boxes(bboxes)
