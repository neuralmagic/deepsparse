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
from deepsparse.timing import InferenceTimingSchema, TimingBuilder


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

        builder.start("total_inference")
        builder.start("pre_process")
        _sleep(s1)
        builder.stop("pre_process")

        builder.start("engine_forward")
        _sleep(s2)
        builder.stop("engine_forward")

        builder.start("post_process")
        _sleep(s3)
        builder.stop("post_process")
        builder.stop("total_inference")

        summary = builder.build()
        timing = InferenceTimingSchema(**summary)

        accuracy = 1.0e-02
        assert timing.pre_process == pytest.approx(s1, accuracy)
        assert timing.engine_forward == pytest.approx(s2, accuracy)
        assert timing.post_process == pytest.approx(s3, accuracy)
        assert timing.total_inference == pytest.approx(s1 + s2 + s3, accuracy)

    def test_always_start_before_complete(self, setup):
        s1, s2, s3, builder = setup

        with pytest.raises(ValueError):
            # builder.start("process") missing
            builder.stop("process")

    def test_never_overwrite(self, setup):
        s1, s2, s3, builder = setup

        builder.start("process")
        _sleep(s1)
        builder.stop("process")
        with pytest.raises(ValueError):
            builder.start("process")
