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

from types import MappingProxyType


__all__ = ["deployment_files"]
"""
Universal container for deployment files
that can be potentially held by any
model
"""

deployment_files = MappingProxyType(
    {
        "ONNX_MODEL_FILE": {"name": "model.onnx"},
        "CONFIG_FILE": {
            "name": "config.json",
            "label_to_class_mapping": "labels_to_class_mapping",
        },
        "TOKENIZER_FILE": {"name": "tokenizer.json"},
        "TOKENIZER_CONFIG_FILE": {"name": "tokenizer_config.json"},
    }
)
