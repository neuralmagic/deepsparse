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

import pytest
from tests.helpers import run_command


@pytest.mark.smoke
def test_check_hardware():
    cmd = ["deepsparse.check_hardware"]
    print(f"\n==== deepsparse.check_hardware command ====\n{' '.join(cmd)}")

    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== deepsparse.check_hardware output ====\n{res.stdout}")

    assert res.returncode == 0, "command exited with non-zero status"
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()

    # check for (static portions of) expected lines
    assert "DeepSparse FP32 model performance supported:" in res.stdout
    assert "DeepSparse INT8 (quantized) model performance supported:" in res.stdout
    assert "Additional CPU info:" in res.stdout
