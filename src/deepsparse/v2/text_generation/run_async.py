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

import asyncio

from deepsparse.transformers.pipelines.text_generation import TextGenerationInput
from deepsparse.v2.text_generation.pipeline import TextGenerationPipeline
from deepsparse.v2.utils import InferenceState


model_path = "hf:mgoin/TinyStories-1M-deepsparse"
pipeline = TextGenerationPipeline(model_path, prompt_sequence_length=3)

prompts = [["Hello there!", "The sun shined bright"], ["The dog barked"]]


async def func(index):
    print("Hello World", index)
    inference_state = InferenceState()
    inference_state.create_state({})
    pipeline_state = pipeline.pipeline_state

    input_value = TextGenerationInput(
     
