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

import pytest
from deepsparse import Pipeline


@pytest.mark.parametrize(
    "pipeline_kwargs",
    [
        dict(
            model_path="zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
            "huggingface/bigpython_bigquery_thepile/base-none",
            engine_type="onnxruntime",
        ),
    ],
)
@pytest.mark.skip(reason="too heavy for now to run in gha")
def test_chat_pipeline_session_manager(pipeline_kwargs):
    chat_pipeline = Pipeline.create(task="chat", **pipeline_kwargs)

    with chat_pipeline.session():
        output_1 = chat_pipeline(
            prompt="first", generation_config=dict(max_new_tokens=1)
        )
        output_2 = chat_pipeline(
            prompt="second", generation_config=dict(max_new_tokens=1)
        )
    # assert inferences in the same context share a session id
    assert output_1.session_ids == output_2.session_ids

    # test that follow-up inference has a different session id
    output_3 = chat_pipeline(prompt="third", generation_config=dict(max_new_tokens=1))
    assert output_3.session_ids != output_1.session_ids
