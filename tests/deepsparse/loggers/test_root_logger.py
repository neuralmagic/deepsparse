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


from unittest.mock import Mock

from deepsparse.loggers.root_logger import RootLogger


def test_log_method():

    mock_leaf_1 = Mock()
    mock_leaf_1.log = Mock()

    mock_leaf_2 = Mock()
    mock_leaf_2.log = Mock()

    mock_leaf_logger = {
        "logger_id_1": mock_leaf_1,
        "logger_id_2": mock_leaf_2,
    }

    mock_config = {
        "tag1": [{"func": "identity", "freq": 2, "uses": ["logger_id_1"]}],
        "tag2": [
            {"func": "identity", "freq": 3, "uses": ["logger_id_2", "logger_id_1"]}
        ],
    }

    root_logger = RootLogger(mock_config, mock_leaf_logger)

    root_logger.log("log_value", "log_type", "tag1")
    assert mock_leaf_1.log.call_count == 0

    root_logger.log("log_value", "log_type", "tag1")
    assert mock_leaf_1.log.call_count == 1

    mock_leaf_logger["logger_id_1"].log.assert_called_with(
        value="log_value",
        tag="tag1",
        func="identity",
        log_type="log_type",
    )

    root_logger.log("log_value", "log_type", "tag2")
    root_logger.log("log_value", "log_type", "tag2")
    assert mock_leaf_2.log.call_count == 0

    root_logger.log("log_value", "log_type", "tag2")
    assert mock_leaf_2.log.call_count == 1
    assert mock_leaf_1.log.call_count == 2

    mock_leaf_logger["logger_id_1"].log.assert_called_with(
        value="log_value",
        tag="tag2",
        func="identity",
        log_type="log_type",
    )

    mock_leaf_logger["logger_id_2"].log.assert_called_with(
        value="log_value",
        tag="tag2",
        func="identity",
        log_type="log_type",
    )
