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

from concurrent.futures import Future
from queue import Queue
from threading import Condition, Lock
from time import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple


__all__ = [
    "ContinuousBatchingQueue",
    "ContinuousBatchingQueues",
    "QueueEntry",
]


# maximum wait time of longest item in queue before it is prioritized
_MAX_WAIT_MS = 100


class QueueEntry(NamedTuple):
    value: Any
    future: Optional[Future]
    entry_time_ms: float

    def time_elapsed(self) -> float:
        return _current_time_ms() - self.entry_time_ms


class ContinuousBatchingQueue(Queue):
    """
    Extension of queue.Queue with helper functions for dequeueing valid
    batch sizes for continuous batching

    :param batch_sizes: valid batch sizes that can be grouped for continuous
        batching
    """

    def __init__(self, batch_sizes: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._batch_sizes = batch_sizes
        self._min_batch_size = min(self.batch_sizes)

    @property
    def batch_sizes(self) -> List[int]:
        """
        :return: valid batch sizes that this queue can return
        """
        return self._batch_sizes

    def pop_batch(self) -> List[Any]:
        """
        :return:
        """
        batch_size = self.max_queued_batch_size()
        if batch_size == 0:
            raise RuntimeError(
                f"Cannot create a batch with {self.qsize()} entries and valid "
                f"batch sizes: {self.batch_sizes}"
            )

        return [self.get() for _ in range(batch_size)]

    def has_batch(self) -> bool:
        """
        :return: True if a batch of valid size can be filled with the current qsize
        """
        return self.qsize() >= self._min_batch_size

    def max_queued_batch_size(self) -> int:
        """
        :return: the maximum batch size that can be filled by members of this queue
        """
        num_entries = self.qsize()
        max_size = 0

        for batch_size in self.batch_sizes:
            if num_entries >= batch_size > max_size:
                # current batch size can be satisfied and is the largest so far
                max_size = batch_size

        return max_size

    def peek(self):
        """
        :return: threadsafe peek of the first item in the queue
        """
        with self.mutex:
            return self.queue[0]


class ContinuousBatchingQueues:
    """
    Threadsafe collection of Queues designed to support continuous batching.
    Each Queue should be keyed by an operator where possible, however keys
    are kept generic.

    On request for next - a job will be returned with an operator key and
    a batch of inputs. The default heuristic for the next job will be
    a combination of wait time and largest batch that can be run
    """

    def __init__(self):
        self._queues = {}  # Dict[Any, ContinuousBatchingQueue]
        self._mutex = Lock()

        # add condition for wait/notify when an item is added to any queue
        self._item_added = Condition(self._mutex)

    def __contains__(self, key: Any) -> bool:
        """
        :param key: key to look up
        :return: True if the given key has a queue in this group
        """
        with self._mutex:
            return key in self._queues

    def add_queue(self, key: Any, batch_sizes: List[int]):
        """
        Adds a queue for a single operator that can be run at multiple batch sizes

        :param key: key to identify queue with, preferably the engine operator
        :param batch_sizes: batch sizes that the operator can be run at
        """
        with self._mutex:
            self._queues[key] = ContinuousBatchingQueue(batch_sizes=batch_sizes)

    def add_queue_item(self, key: Any, item: Any, future: Optional[Future] = None):
        """
        Adds an item to the given queue

        :param key: key for queue to add to
        :param item: item to add in queue
        :param future: optional future that should be used for resolution of value
        """
        if key not in self:
            raise KeyError(f"Cannot add item to queue for unregistered key {key}")

        entry = QueueEntry(value=item, future=future, entry_time_ms=_current_time_ms())

        with self._mutex:
            self._queues[key].put(entry)
            self._item_added.notify()

    def has_next_batch(self) -> bool:
        """
        :return: true if any Queue has enough entries to fill a valid batch size
        """
        with self._mutex:
            return any(queue.has_batch() for queue in self._queues.values())

    def pop_batch(
        self,
        select_fn: Callable[[Dict[Any, ContinuousBatchingQueue]], Any] = None,
        block: bool = True,
    ) -> Tuple[Any, List[QueueEntry]]:
        """
        :param select_fn: function that takes in a dictionary of queue key
            (i.e. EngineOperator) to its ContinuousBatchingQueue of QueueItem
            objects and returns the key of the queue that should be returned.
            Only keys with queues large enough to fill a batch will be given.
            If not provided, the default select_fn will return the queue that
            can fill the largest batch size, or the queue that has the first item
            with the longest wait time if that time is over 100ms.
        :param block: if True, will wait for a valid batch to be in a queue before
            popping and returning, if False, will raise an error if a full batch
            cannot be popped. Default True
        :return: Tuple of the queue key (EngineOperator) and
            batch of QueueEntry objects as a list that have been popped and should
            be run as a batch
        """
        with self._mutex:
            while not (valid_queues := self._filter_empty_queues()):
                if block:
                    # wait to search for a valid queue again until a new item is added
                    self._item_added.wait()
                else:
                    raise RuntimeError(
                        "Cannot pop_batch when no queues have enough items to fill "
                        "a valid batch size, check with has_next_batch before calling "
                        "pop_batch"
                    )

            select_fn = select_fn or _default_select_fn
            selected_key = select_fn(valid_queues)

            return selected_key, self._queues[selected_key].pop_batch()

    def _filter_empty_queues(self) -> Dict[Any, ContinuousBatchingQueue]:
        return {key: queue for key, queue in self._queues.items() if queue.has_batch()}


def _default_select_fn(queues: Dict[Any, ContinuousBatchingQueue]) -> Any:
    # find the maximum wait time of a queue
    wait_times = [(key, queue.peek().time_elapsed()) for key, queue in queues.items()]
    max_wait_key, max_wait = max(wait_times, key=lambda x: x[1])  # key on time

    if max_wait >= _MAX_WAIT_MS:
        # if max time is greater than the threshold return that queue
        return max_wait_key

    # default to the largest batch size that can be satisfied
    return max(queues.keys(), key=lambda key: queues[key].max_queued_batch_size())


def _current_time_ms():
    return time() * 1000
