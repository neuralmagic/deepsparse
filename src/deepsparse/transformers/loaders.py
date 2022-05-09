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

"""
Utilities for loading batches from files for pipelines
"""

import json
from abc import ABC, abstractmethod
from csv import DictReader
from pathlib import Path
from typing import Any, Dict, List, Optional


__all__ = [
    "get_batch_loader",
    "SUPPORTED_EXTENSIONS",
]

SUPPORTED_EXTENSIONS = [".json", ".csv", ".txt"]


class _BatchLoader(ABC):
    # Base class for all BatchLoaders
    def __init__(self, data_file: str, batch_size: int = 1):
        self.data_file = data_file
        self.batch_size = batch_size
        self.header = None

    @abstractmethod
    def _get_reader(self, filename) -> List[Dict[str, str]]:
        raise NotImplementedError

    def add_to_batch(
        self,
        input_sample: Dict[str, Any],
        batch: Optional[Dict[str, List[Any]]],
    ) -> Dict[str, List[Any]]:
        """
        Add dict type input to batch
        Note: Updates the header with keys of input_sample, if batch evaluates
        to False.

        :param input_sample: A dict representing one sample
        :param batch: A dict with same keys as input
        :return: items of input_sample appended to the right keys
        """
        if not batch:
            self.header = list(input_sample.keys())
            batch = {key: [input_sample[key]] for key in self.header}
        else:
            for key in self.header:
                batch[key].append(input_sample[key])
        return batch

    def pad_last_batch(self, batch):
        """
        Pads the batch with last added value
        Batch must have at-least one value

        :param batch: The batch to be padded to batch_size
        :return: The padded batch
        """
        if self.header and 0 < len(batch[self.header[0]]) < self.batch_size:
            for key in self.header:
                repeat_element = batch[key][-1]
                copies_needed = self.batch_size - len(batch[key])
                extra_elements = [repeat_element] * copies_needed
                batch[key].extend(extra_elements)

            yield batch

    def __iter__(self) -> Optional[Dict[str, Any]]:
        # Note: json file should contain one json object per line
        batch = None
        with open(self.data_file) as _input_file:
            for _input in self._get_reader(_input_file):
                batch = self.add_to_batch(_input, batch)
                if len(batch[self.header[0]]) == self.batch_size:
                    yield batch
                    batch = {key: [] for key in batch}
        yield from self.pad_last_batch(batch)


class _JSONBatchLoader(_BatchLoader):
    # Convenience class to read batches from JSON files

    def _get_reader(self, filename) -> List[Dict[str, str]]:
        return (json.loads(line) for line in filename)


class _CSVBatchLoader(_BatchLoader):
    # Convenience class to read batches from CSV files

    def _get_reader(self, filename) -> List[Dict[str, str]]:
        return DictReader(filename)


class _TextBatchLoader(_BatchLoader):
    # Convenience class to read batches from TEXT files
    # Note: Does not support Question-Answering task

    def __init__(self, data_file: str, batch_size: int = 1, task: str = None):
        super().__init__(data_file=data_file, batch_size=batch_size)
        task = task.lower().replace("_", "-") if task else ""
        if task in ["ner", "token-classification"]:
            self.header = ["inputs"]
        elif task in ["sentiment-analysis", "text-classification"]:
            self.header = ["sequences"]
        else:
            raise ValueError(f"{task} does not support text file as input")

    def _get_reader(self, filename) -> List[Dict[str, str]]:
        return ({self.header[0]: line.strip()} for line in filename)


def get_batch_loader(
    data_file: str, batch_size: int = 1, task: str = None
) -> _BatchLoader:
    """
    Returns the corresponding BatchLoader based on filetype

    :param data_file: The file to read data from, can be json, csv, or txt
    :param batch_size: The batch size to use for creating batches
    :param task: The task, batches are to be generated for
    :return: Respective _BatchLoader object based on filetype
    """

    data_file_obj = Path(data_file)
    file_type = data_file_obj.suffix

    if file_type == ".json":
        return _JSONBatchLoader(data_file=data_file, batch_size=batch_size)

    if file_type == ".csv":
        return _CSVBatchLoader(data_file=data_file, batch_size=batch_size)

    if file_type == ".txt":
        return _TextBatchLoader(data_file=data_file, batch_size=batch_size, task=task)

    raise ValueError(f"input file {data_file} not supported")
