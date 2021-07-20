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
import os
from typing import Generator, Iterable, List, Tuple, Union

import numpy

from sparseml.utils import load_numpy
from sparsezoo import Zoo
from sparsezoo.utils import load_numpy_list


"""
A file with utility functions and helper classes for loading data
"""


class _BatchLoader:
    # Helper class for loading batches
    __slots__ = [
        "_data",
        "_batch_size",
        "_single_input",
        "_iterations",
        "_batch_buffer",
        "_batch_template",
        "_batches_created",
    ]

    def __init__(
        self,
        data: Iterable[Union[numpy.ndarray, List[numpy.ndarray]]],
        batch_size: int,
        iterations: int,
    ):
        self._data = data
        self._single_input = type(self._data[0]) is numpy.ndarray
        if self._single_input:
            self._data = [self._data]
        self._batch_size = batch_size
        self._iterations = iterations
        if batch_size < 0 or iterations < 0:
            raise ValueError(
                f"Both batch size and number of iterations should be non-negative, "
                f"supplied values (batch_size, iterations):{(batch_size, iterations)}"
            )

        self._batch_buffer = []
        self._batch_template = self._init_batch_template()
        self._batches_created = 0

    def __iter__(self) -> Generator[List[numpy.ndarray], None, None]:
        yield from self._multi_input_batch_generator()

    @property
    def _buffer_is_full(self) -> bool:
        return len(self._batch_buffer) == self._batch_size

    @property
    def _all_batches_loaded(self) -> bool:
        return self._batches_created >= self._iterations

    def _multi_input_batch_generator(
        self,
    ) -> Generator[List[numpy.ndarray], None, None]:
        # A generator for with each element of the form
        # [[(batch_size, features_a), (batch_size, features_b), ...]]
        while not self._all_batches_loaded:
            yield from self._batch_generator(source=self._data)

    def _batch_generator(self, source) -> Generator[List[numpy.ndarray], None, None]:
        # batches from source
        for sample in source:
            self._batch_buffer.append(sample)
            if self._buffer_is_full:
                _batch = self._make_batch()
                yield _batch
                self._batch_buffer = []
                self._batches_created += 1
                if self._all_batches_loaded:
                    break

    def _init_batch_template(
        self,
    ) -> Iterable[Union[List[numpy.ndarray], numpy.ndarray]]:

        # A placeholder for batches

        return [
            numpy.ascontiguousarray(
                numpy.zeros((self._batch_size, *_input.shape), dtype=_input.dtype)
            )
            for _input in self._data[0]
        ]

    def _make_batch(self) -> Iterable[Union[numpy.ndarray, List[numpy.ndarray]]]:
        # Copy contents of buffer to batch placeholder
        # and return A list of numpy array(s) representing the batch

        batch = [
            numpy.stack([sample[idx] for sample in self._batch_buffer], out=template)
            for idx, template in enumerate(self._batch_template)
        ]

        if self._single_input:
            batch = batch[0]
        return batch


def load_data(
    data_path: str,
) -> List[List[numpy.ndarray]]:
    """
    Loads data from given sparseZoo stub or directory with .npz files
    :param data_path: directory path to .npz files to load or SparseZoo stub
    :return: List of loaded npz files
    """

    if data_path.startswith("zoo:"):
        data_dir = Zoo.load_model_from_stub(data_path).data_inputs.downloaded_path()
    else:
        data_dir = data_path
        data_files = os.listdir(data_dir)
        if any(".npz" not in file_name for file_name in data_files):
            raise RuntimeError(
                f"All files in data directory {data_dir} must have a .npz extension "
                f"found {[name for name in data_files if '.npz' not in name]}"
            )

    samples = load_numpy_list(data_dir)
    # unwrap unloaded numpy files
    samples = [
        load_numpy(sample) if isinstance(sample, str) else sample for sample in samples
    ]

    processed_samples = []
    for idx, sample in enumerate(samples):
        sample = list(sample.values())
        processed_samples.append(sample)

    return processed_samples


def _load_random_data(
    batch_size: int, iterations: int, image_size: Tuple[int]
) -> Generator[List[numpy.ndarray], None, None]:
    _channels = 3
    iteration = 0

    while iteration < iterations:
        yield [
            (numpy.random.rand(*(batch_size, _channels, *image_size))).astype(
                numpy.float32
            )
        ]
        iteration += 1
