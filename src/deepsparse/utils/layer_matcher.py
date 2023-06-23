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


class LayerMatcher:
    def __init__(onnx_model, torch_model):
        self._onnx = onnx_model
        self._torch = torch_model

        self._validate()

    def _validate(self):
        """
        validate the input models
        """
        pass

    @lru_cache
    def match(self):
        """
        output
        {
            "torch
        }
        """
        pass

    def map_onnx_to_torch_names(onn_names_to_weights: Dict[str, numpy.ndarray]):
        pass

    def map_onnx_to_torch_names(
        self,
        onnx_names_to_weights: Dict[str, numpy.ndarray],
        torch_names_to_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, str]:

        """
        {
            onnx_names : weight
        }
        {
            torch_names : weight
        }

        ->

        {
            torch_names : onnx_names
        }

        """
        pass

    def mismatch(self):
        """
        show the mismatches
        """
        pass
