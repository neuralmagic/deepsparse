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

import time
from threading import Thread

import pytest
from deepsparse.schedulers.utils import (
    ContinuousBatchingQueue,
    ContinuousBatchingQueues,
    QueueEntry,
)


@pytest.mark.parametrize(
    "batch_sizes,num_entries,expected_batch_size",
    [
        ([1, 4, 8], 20, 8),
        ([1, 4, 8], 6, 4),
        ([1, 4, 8], 4, 4),
        ([1, 4, 8], 3, 1),
        ([4], 5, 4),
    ],
)
@pytest.mark.skip("debuging")
def test_queue_single_pop(batch_sizes, num_entries, expected_batch_size):
    queue = ContinuousBatchingQueue(batch_sizes=batch_sizes)
    assert not queue.has_batch()
    for i in range(num_entries):
        queue.put(i)

    assert queue.has_batch()
    assert queue.max_queued_batch_size() == expected_batch_size

    batch = queue.pop_batch()
    assert len(batch) == expected_batch_size
    assert batch == list(range(expected_batch_size))


@pytest.mark.skip("debuging")
def test_queue_multi_pop():
    queue = ContinuousBatchingQueue(batch_sizes=[2, 4, 8])

    for i in range(23):
        if i < 2:
            assert not queue.has_batch()
        else:
            assert queue.has_batch()
        queue.put(i)

    def pop_and_assert_queue_size_and_pop(expected_qsize, expected_batch_size):
        assert queue.qsize() == expected_qsize
        assert queue.has_batch()
        assert queue.max_queued_batch_size() == expected_batch_size
        assert len(queue.pop_batch()) == expected_batch_size

    # pop items from queue, checkign remaining qsize and correct batch size is popped
    pop_and_assert_queue_size_and_pop(23, 8)
    pop_and_assert_queue_size_and_pop(15, 8)
    pop_and_assert_queue_size_and_pop(7, 4)
    pop_and_assert_queue_size_and_pop(3, 2)

    assert not queue.has_batch()
    queue.put(23)
    pop_and_assert_queue_size_and_pop(2, 2)

    assert queue.empty()


@pytest.mark.skip("debuging")
def test_queue_invalid_pop():
    queue = ContinuousBatchingQueue(batch_sizes=[4, 8])
    for i in range(3):
        queue.put(i)

    with pytest.raises(RuntimeError):
        # queue size 3, min batch size 4
        queue.pop_batch()


@pytest.mark.skip("debuging")
def test_queues_pop_batch_max_valid_batch():
    queues = ContinuousBatchingQueues()

    queues.add_queue("key_1", [2, 4])
    queues.add_queue("key_2", [3])

    assert not queues.has_next_batch()

    queues.add_queue_item("key_1", 1)
    queues.add_queue_item("key_1", 2)
    assert queues.has_next_batch()

    queues.add_queue_item("key_2", 1)
    queues.add_queue_item("key_2", 2)
    queues.add_queue_item("key_2", 3)
    # NOTE - if this block takes more than 100ms, test may fail
    # as timeout may lead key_1 to be popped first

    # key_2 should be popped first because it has larger loaded batch size
    first_popped_key, first_popped_batch = queues.pop_batch()
    assert first_popped_key == "key_2"
    assert len(first_popped_batch) == 3
    assert all(isinstance(item, QueueEntry) for item in first_popped_batch)

    assert queues.has_next_batch()

    second_popped_key, second_popped_batch = queues.pop_batch()
    assert second_popped_key == "key_1"
    assert len(second_popped_batch) == 2
    assert all(isinstance(item, QueueEntry) for item in second_popped_batch)


@pytest.mark.skip("debuging")
def test_queues_pop_batch_time_elapsed_priority():
    queues = ContinuousBatchingQueues()

    queues.add_queue("key_1", [2, 4])
    queues.add_queue("key_2", [3])

    assert not queues.has_next_batch()

    queues.add_queue_item("key_1", 1)
    queues.add_queue_item("key_1", 2)
    assert queues.has_next_batch()

    # sleep 150ms (time threshold is 100ms)
    time.sleep(0.15)

    queues.add_queue_item("key_2", 1)
    queues.add_queue_item("key_2", 2)
    queues.add_queue_item("key_2", 3)

    # key 1 should be popped first because its first item has been waiting longer
    # than the time threshold and key_2 was just added

    popped_key, popped_batch = queues.pop_batch()
    assert popped_key == "key_1"
    assert len(popped_batch) == 2


@pytest.mark.skip("debuging")
def test_queues_pop_batch_blocking():
    queues = ContinuousBatchingQueues()
    queues.add_queue("key_1", [2])

    def test_fn():
        # pop batch and block until true
        key, batch = queues.pop_batch(block=True)
        # compare to expected results
        assert key == "key_1"
        assert batch == [1, 2]

    # start a thread to pop batch
    # it should hang indefinitely because block=True and there are no items yet in queue
    thread = Thread(target=queues.pop_batch)
    thread.start()

    # confirm thread is still running
    assert thread.is_alive()
    time.sleep(0.15)
    # sleep and confirm thread is still hanging
    assert thread.is_alive()

    # confirm thread still runs after a single insertion (min batch size is 2)
    queues.add_queue_item("key_1", 1)
    assert thread.is_alive()

    # add a second item and assert thread finishes
    queues.add_queue_item("key_1", 2)
    time.sleep(0.1)
    assert not thread.is_alive()
