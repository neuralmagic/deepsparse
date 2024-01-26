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

import yaml

from deepsparse.loggers.config import LoggerConfig, LoggingConfig


def test_config_generates_default_json():
    """Check the default LoggingConfig"""

    expected_config = """
    loggers:
      default:
        name: PythonLogger
        handler: null # None in python
    system:
      "re:.*":
      - func: identity
        freq: 1
        uses:
        - default
    performance:
      cpu:
      - func: identity
        freq: 1
        uses:
        - default
    metric:
      "(?i)operator":
      - func: identity
        freq: 1
        uses:
        - default
        capture: null

    """
    expected_dict = yaml.safe_load(expected_config)
    default_dict = LoggingConfig().dict()
    assert expected_dict == default_dict


def test_logger_config_accepts_kwargs():
    expected_config = """
    name: PythonLogger
    foo: 1
    bar: "2024"
    baz:
      one: 1
      two: 2
    boston:
      - one
      - two
    """
    config = LoggerConfig(**yaml.safe_load(expected_config)).dict()

    assert config["name"] == "PythonLogger"
    assert config["handler"] is None
    assert config["baz"] == dict(one=1, two=2)
    assert config["foo"] == 1
    assert config["boston"] == ["one", "two"]
    assert config["bar"] == "2024"
