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
from typing import Dict, List

import pytest
from tests.helpers import run_command


@pytest.mark.smoke
def test_yolov8_annotate(cleanup: Dict[str, List]):
    cmd = [
        "deepsparse.yolov8.annotate",
        "--source",
        "sample_images/basilica.jpg",
        "--model_filepath",
        "zoo:yolov8-n-coco-base_quantized",
    ]
    expected_output_path = "annotation-results/deepsparse-annotations/result-0.jpg"

    cleanup["files"].append(expected_output_path)
    print(f"\n==== test_yolov8_annotate command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_yolov8_annotate output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()

    # check output file exists
    assert os.path.exists(expected_output_path)
