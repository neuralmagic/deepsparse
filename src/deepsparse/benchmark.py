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
Code related to benchmarking batched inference runs
"""

from typing import Any, Dict, Iterable, Iterator, List, Union

import numpy


__all__ = ["BatchBenchmarkResult", "BenchmarkResults"]


class BatchBenchmarkResult(object):
    """
    A benchmark result for a batched inference run

    :param time_start: The system time when the run for the batch was started
    :param time_end: The system time when the run for the batch ended
    :param batch_size: The size of the batch that was benchmarked
    :param inputs: Optional batch inputs that were given for the run
    :param outputs: Optional batch outputs that were given for the run
    :param extras: Optional batch extras to store any other data for the run
    """

    def __init__(
        self,
        time_start: float,
        time_end: float,
        batch_size: int,
        inputs: Union[None, List[numpy.ndarray]] = None,
        outputs: Union[None, List[numpy.ndarray], Dict[str, numpy.ndarray]] = None,
        extras: Any = None,
    ):
        self._time_start = time_start
        self._time_end = time_end
        self._batch_size = batch_size
        self._inputs = inputs
        self._outputs = outputs
        self._extras = extras

    def __repr__(self):
        props = {
            "time_start": self.time_start,
            "time_end": self.time_end,
            "size": self.batch_size,
            "batches_per_second": self.batches_per_second,
            "items_per_second": self.items_per_second,
            "ms_per_batch": self.ms_per_batch,
            "ms_per_item": self.ms_per_item,
        }

        return f"{self.__class__.__name__}({props})"

    def __str__(self):
        return (
            f"{self.__class__.__name__}(ms_per_batch={self.ms_per_batch}, "
            f"items_per_second={self.items_per_second})"
        )

    @property
    def time_start(self) -> float:
        """
        :return: The system time when the run for the batch was started
        """
        return self._time_start

    @property
    def time_end(self) -> float:
        """
        :return: The system time when the run for the batch ended
        """
        return self._time_end

    @property
    def time_elapsed(self) -> float:
        """
        :return: The time elapsed for the entire run (end - start)
        """
        return self._time_end - self._time_start

    @property
    def batch_size(self) -> int:
        """
        :return: The size of the batch that was benchmarked
        """
        return self._batch_size

    @property
    def inputs(self) -> Union[None, List[numpy.ndarray]]:
        """
        :return: Batch inputs that were given for the run, if any
        """
        return self._inputs

    @property
    def outputs(self) -> Union[None, List[numpy.ndarray]]:
        """
        :return: Batch outputs that were given for the run, if any
        """
        return self._outputs

    @property
    def extras(self) -> Any:
        """
        :return: Batch extras to store any other data for the run
        """
        return self._extras

    @property
    def batches_per_second(self) -> float:
        """
        :return: The number of batches that could be run in one second
            based on this result
        """
        return 1.0 / self.time_elapsed

    @property
    def items_per_second(self) -> float:
        """
        :return: The number of items that could be run in one second
            based on this result
        """
        return self._batch_size / self.time_elapsed

    @property
    def ms_per_batch(self) -> float:
        """
        :return: The number of milliseconds it took to run the batch
        """
        return self.time_elapsed * 1000.0

    @property
    def ms_per_item(self) -> float:
        """
        :return: The averaged number of milliseconds it took to run each item
            in the batch
        """
        return self.time_elapsed * 1000.0 / self._batch_size


class BenchmarkResults(Iterable):
    """
    The benchmark results for a list of batched inference runs
    """

    def __init__(self):
        self._results = []  # type: List[BatchBenchmarkResult]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._properties_dict})"

    def __str__(self):
        """
        :return: Human readable form of the benchmark summary
        """
        formatted_props = [
            "\t{}: {}".format(key, val) for key, val in self._properties_dict.items()
        ]
        return "{}:\n{}".format(
            self.__class__.__name__,
            "\n".join(formatted_props),
        )

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, index: int) -> BatchBenchmarkResult:
        return self._results[index]

    def __iter__(self) -> Iterator[BatchBenchmarkResult]:
        for result in self._results:
            yield result

    @property
    def _properties_dict(self) -> Dict:
        return {
            "items_per_second": self.items_per_second,
            "ms_per_batch": self.ms_per_batch,
            "batch_times_mean": self.batch_times_mean,
            "batch_times_median": self.batch_times_median,
            "batch_times_std": self.batch_times_std,
        }

    @property
    def results(self) -> List[BatchBenchmarkResult]:
        """
        :return: the list of recorded batch results
        """
        return self._results

    @property
    def num_batches(self) -> int:
        """
        :return: the number of batches that have been added
        """
        return len(self)

    @property
    def num_items(self) -> int:
        """
        :return: the number of items across all batches that have been added
        """
        num_items = sum([res.batch_size for res in self._results])

        return num_items

    @property
    def batch_times(self) -> List[float]:
        """
        :return: the list of all batch run times that have been added
        """
        return [res.time_elapsed for res in self._results]

    @property
    def batch_sizes(self) -> List[int]:
        """
        :return: the list of all batch run sizes that have been added
        """
        return [res.batch_size for res in self._results]

    @property
    def batch_times_mean(self) -> float:
        """
        :return: the mean of all the batch run times that have been added
        """
        return numpy.mean(self.batch_times).item()

    @property
    def batch_times_median(self) -> float:
        """
        :return: the median of all the batch run times that have been added
        """
        return numpy.median(self.batch_times).item()

    @property
    def batch_times_std(self) -> float:
        """
        :return: the standard deviation of all the batch run times that have been added
        """
        return numpy.std(self.batch_times).item()

    @property
    def batches_per_second(self) -> float:
        """
        :return: The number of batches that could be run in one second
            based on this result
        """
        return self.num_batches / sum(self.batch_times)

    @property
    def items_per_second(self) -> float:
        """
        :return: The number of items that could be run in one second
            based on this result
        """
        return self.num_items / sum(self.batch_times)

    @property
    def ms_per_batch(self) -> float:
        """
        :return: The number of milliseconds it took to run the batch
        """
        return sum(self.batch_times) * 1000.0 / self.num_batches

    @property
    def ms_per_item(self) -> float:
        """
        :return: The averaged number of milliseconds it took to run each item
            in the batch
        """
        return sum(self.batch_times) * 1000.0 / self.num_items

    @property
    def inputs(self) -> Union[None, List[numpy.ndarray]]:
        """
        :return: Batch inputs that were given for the run, if any
        """
        return [res.inputs for res in self._results]

    @property
    def outputs(self) -> Union[None, List[numpy.ndarray]]:
        """
        :return: Batch outputs that were given for the run, if any
        """
        return [res.outputs for res in self._results]

    def append_batch(
        self,
        time_start: float,
        time_end: float,
        batch_size: int,
        inputs: Union[None, List[numpy.ndarray]] = None,
        outputs: Union[None, List[numpy.ndarray], Dict[str, numpy.ndarray]] = None,
        extras: Any = None,
    ):
        """
        Add a recorded batch to the current results

        :param time_start: The system time when the run for the batch was started
        :param time_end: The system time when the run for the batch ended
        :param batch_size: The size of the batch that was benchmarked
        :param inputs: Optional batch inputs that were given for the run
        :param outputs: Optional batch outputs that were given for the run
        :param extras: Optional batch extras to store any other data for the run
        """
        self._results.append(
            BatchBenchmarkResult(
                time_start, time_end, batch_size, inputs, outputs, extras
            )
        )
