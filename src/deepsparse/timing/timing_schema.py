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

from typing import Iterable

from pydantic import BaseModel


class InferenceTimingSchema(BaseModel):
    """
    Stores the information about time deltas
    (in seconds) of certain processes within
    the inference pipeline
    """

    pre_process_delta: float
    engine_forward_delta: float
    post_process_delta: float
    total_inference_delta: float

    @classmethod
    def aggregate(
        cls,
        batched_inference_timing: Iterable["InferenceTimingSchema"],
        consolidation_func=sum,
    ) -> "InferenceTimingSchema":
        """
        Aggregates (merges) a batch of inference timing pydantic
        models according to the `consolidation_func` function.

        :param batched_inference_timing: A batch of inference timing
            pydantic models
        :param consolidation_func: Function that acts along the fields
            of pydantic models
        :return: A single, aggregated inference timing pydantic model
        """
        # translate Pydantic model to dictionary for easier manipulation
        batched_inference_timing = [dict(timing) for timing in batched_inference_timing]
        single_batch = batched_inference_timing[0]
        field_names = single_batch.keys()

        aggregated_fields = {}
        for field_name in field_names:
            # aggregate every field
            aggregated_fields[field_name] = consolidation_func(
                timing[field_name] for timing in batched_inference_timing
            )

        return cls(**aggregated_fields)
