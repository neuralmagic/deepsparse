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
from deepsparse.timing import TimingBuilder


def _sleep(sleep_time):
    time.sleep(sleep_time)


@pytest.mark.parametrize("s1, s2, s3", [(0.1, 0.2, 0.3)])
class TestTimingBuilder:
    @pytest.fixture
    def setup(self, s1, s2, s3):
        builder = TimingBuilder()
        yield s1, s2, s3, builder

    def test_happy_pathway(self, setup):
        s1, s2, s3, builder = setup
        builder.start()

        builder.pre_process_start()
        _sleep(s1)
        builder.pre_process_complete()

        builder.engine_forward_start()
        _sleep(s2)
        builder.engine_forward_complete()

        builder.post_process_start()
        _sleep(s3)
        builder.post_process_complete()

        summary = builder.build()

        assert builder.t0 is None
        assert builder.t1 is None

        accuracy = 1.0e-02
        assert summary.pre_process_delta == pytest.approx(s1, accuracy)
        assert summary.engine_forward_delta == pytest.approx(s2, accuracy)
        assert summary.post_process_delta == pytest.approx(s3, accuracy)
        assert summary.total_inference_delta == pytest.approx(s1 + s2 + s3, accuracy)

    def test_always_start(self, setup):
        s1, s2, s3, builder = setup

        with pytest.raises(ValueError):
            # builder.start() missing
            builder.pre_process_start()

    def test_always_start_before_complete(self, setup):
        s1, s2, s3, builder = setup

        with pytest.raises(ValueError):
            # builder.pre_process_start() missing
            builder.pre_process_complete()

    def test_never_start_twice(self, setup):
        s1, s2, s3, builder = setup

        builder.start()
        with pytest.raises(ValueError):
            builder.start()

    def test_never_overwrite(self, setup):
        s1, s2, s3, builder = setup
        builder.start()

        builder.pre_process_start()
        _sleep(s1)
        builder.pre_process_complete()
        with pytest.raises(ValueError):
            # builder.build() missing
            builder.pre_process_start()
