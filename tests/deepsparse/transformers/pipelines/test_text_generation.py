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

from transformers import AutoModelForCausalLM, AutoTokenizer

import pytest
from deepsparse import Pipeline


@pytest.mark.slow
def test_codegen():
    model_stub = (
        "zoo:nlg/text_generation/codegen_mono-350m/pytorch/"
        "huggingface/bigpython_bigquery_thepile/base-none"
    )
    pipeline = Pipeline.create(
        task="text_generation",
        model_path=model_stub,
        sequence_length=16,
        max_generated_tokens=8,
        prompt_processing_sequence_length=1,
        use_deepsparse_cache=False,
    )
