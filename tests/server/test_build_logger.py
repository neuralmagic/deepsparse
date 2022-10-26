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

from deepsparse.server.build_logger import build_logger


YAML_CONFIG_1 = ""


def test_build_logger(yaml_config, expected_logger):
    server_config = ServerConfig(*yaml_config)
    logger = build_logger(server_config)
    expected_logger = None
    assert logger == expected_logger
