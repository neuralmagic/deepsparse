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

from unittest.mock import patch

import numpy as np

from deepsparse.transformers.engines import NLDecoderEngine
from flaky import flaky


@flaky(max_runs=10, min_passes=1)
def test_generate_token():
    logits = np.array([1.0, 11, 0.9, 0.8])
    expected_token = 1

    with patch.object(NLDecoderEngine, "__init__", lambda x, y, z: None):
        engine = NLDecoderEngine(None, None)
        engine.deterministic = False
        engine.sampling_temperature = 1.0
        token = engine.generate_token(logits)

    assert expected_token == token
