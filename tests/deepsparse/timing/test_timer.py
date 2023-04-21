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

import pytest
from deepsparse.timing import Timer


def _sleep(sleep_time):
    time.sleep(sleep_time)


@pytest.mark.parametrize("s1, s2, s3", [(0.1, 0.2, 0.3)])
class TestTimer:
    @pytest.fixture
    def setup(self, s1, s2, s3):
        timer = Timer()
        yield s1, s2, s3, timer

    def test_happy_pathway(self, setup):
        s1, s2, s3, timer = setup

        timer.start("total_inference")
        timer.start("pre_process")
        _sleep(s1)
        timer.stop("pre_process")

        pre_process_time_delta = timer.time_delta("pre_process")

        timer.start("engine_forward")
        _sleep(s2)
        timer.stop("engine_forward")

        timer.start("post_process")
        _sleep(s3)
        timer.stop("post_process")
        timer.stop("total_inference")

        engine_forward_time_delta = timer.time_delta("engine_forward")
        post_process_time_delta = timer.time_delta("post_process")
        total_inference_time_delta = timer.time_delta("total_inference")

        accuracy = 1.0e-02
        assert pre_process_time_delta == pytest.approx(s1, accuracy)
        assert engine_forward_time_delta == pytest.approx(s2, accuracy)
        assert post_process_time_delta == pytest.approx(s3, accuracy)
        assert total_inference_time_delta == pytest.approx(s1 + s2 + s3, accuracy)

    def test_always_start_before_complete(self, setup):
        _, _, _, timer = setup

        with pytest.raises(ValueError):
            # builder.start("process") missing
            timer.stop("process")

    def test_never_overwrite(self, setup):
        s1, _, _, timer = setup

        timer.start("process")
        _sleep(s1)
        timer.stop("process")
        with pytest.raises(ValueError):
            timer.start("process")

    def test_cannot_compute_time_delta_before_start(self, setup):
        _, _, _, timer = setup

        with pytest.raises(ValueError):
            timer.time_delta("process")

    def test_cannot_compute_time_delta_before_stop(self, setup):
        s1, _, _, timer = setup

        timer.start("process")
        _sleep(s1)
        with pytest.raises(ValueError):
            timer.time_delta("process")
