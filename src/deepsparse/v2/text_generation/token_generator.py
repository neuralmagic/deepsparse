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
from deepsparse.transformers.utils.token_generator import TokenGenerator
from deepsparse.v2.operators import Operator


__all__ = ["TokenGeneratorOperator"]


class TokenGeneratorOperator(Operator):
    def run(self, logits_shape, deterministic, tokens, sampling_temperature, **kwargs):
        token_generator = TokenGenerator(
            logits_shape=logits_shape,
            deterministic=deterministic,
            tokens=tokens,
            sampling_temperature=sampling_temperature,
            **kwargs,
        )
        return {"token_generator": token_generator}
