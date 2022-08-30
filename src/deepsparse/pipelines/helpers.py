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

from dataclasses import dataclass


__all__ = ["DeploymentFiles"]


@dataclass(frozen=True)
class DeploymentFiles:
    """
    Universal container for deployment files
    that can be potentially held by any
    model. Additionally, holds any
    additional keys pertaining to the
    deployment files
    """

    ONNX_MODEL_FILE: str = "model.onnx"
    CONFIG_FILE: str = "config.json"
    TOKENIZER_FILE: str = "tokenizer.json"
    TOKENIZER_CONFIG_FILE: str = "tokenizer_config.json"
    LABEL_TO_CLASS_MAPPING: str = "label_to_class_mapping"
