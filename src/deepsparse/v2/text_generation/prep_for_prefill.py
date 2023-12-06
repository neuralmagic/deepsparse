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

import logging
from typing import Any

from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import PipelineState


_LOGGER = logging.getLogger(__name__)

__all__ = ["PrepareforPrefill"]


class PrepareforPrefill(Operator):
    def __init__(self, kv_cache_creator: Operator):
        """
        Operator before prefill. Responsible for creating the kv_cache based on engine
        variables. Currently, this operator expects that the kv_cache_creator is
        provided during initization and then uses pipeline_state to run the
        kv_cache_operator.
        """
        # NOTE: Alternatively, we can initialize the kv_cache_creater operator here,
        # instead of at the pipeline level.
        self.kv_cache_creator = kv_cache_creator

        _LOGGER.warn(
            "This operator requires the PipelineState to be set-up with the "
            "cache_shape, output_names, kv_cache_data_type attributes to be set "
            "from the NLEngineOperator"
        )

    def run(
        self,
        input_ids: Any,
        attention_mask: Any,
        pipeline_state: PipelineState,
        **kwargs,
    ):
        # NOTE: Can potentially just be class attributes instead of relying on
        # pipeline state.
        cache_shape = pipeline_state.current_state.get("cache_shape")
        data_type = pipeline_state.current_state.get("kv_cache_data_type")
        output_names = pipeline_state.current_state.get("output_names")

        tokens = input_ids[attention_mask.nonzero()].tolist()
        kv_cache = self.kv_cache_creator.run(
            cache_shape=cache_shape,
            kv_cache_data_type=data_type,
            output_names=output_names,
        ).get("kv_cache")
        return {"tokens": tokens, "kv_cache": kv_cache}
