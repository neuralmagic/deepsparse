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

from typing import Any

import numpy
from pydantic import BaseModel

from deepsparse.operators.engine_operator import EngineOperator, EngineOperatorInputs
from deepsparse.transformers.helpers import overwrite_transformer_onnx_model_inputs


__all__ = [
    "NLEngineOperatorNoCache",
    "NLEngineInputsNoCache",
]


class NLEngineInputsNoCache(BaseModel):
    input_ids: Any = None
    attention_mask: Any = None


class NLEngineOperatorNoCache(EngineOperator):
    """
    Operator the Natural Language Engine, that operates without
    KV Cache. This means that this operator merely maps input_ids
    and attention_mask to logits
    """

    input_schema = NLEngineInputsNoCache
    output_schema = None

    def __init__(self, sequence_length: int, **kwargs):
        overwrite_transformer_onnx_model_inputs(
            path=kwargs.get("model_path"),
            batch_size=kwargs.get("batch_size", 1),
            max_length=sequence_length,
        )
        super().__init__(**kwargs)

    def run(self, inp: NLEngineInputsNoCache, **kwargs) -> Any:
        engine_inputs = [inp.input_ids, inp.attention_mask]
        logits = (
            super()
            .run(EngineOperatorInputs(engine_inputs=engine_inputs), **kwargs)
            .get("engine_outputs")
        )

        # By default, the engine outputs logits for all tokens in the sequence.
        # Let's filter out the logits for the padding tokens.
        logits = numpy.compress(inp.attention_mask.flatten(), logits[0], axis=1)

        return {"logits": [logits], "kv_cache": None, "tokens": None}, {
            "prompt_logits": [logits]
        }
