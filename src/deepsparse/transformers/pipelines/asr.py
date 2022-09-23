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

from pydantic import BaseModel, Field
from transformers import Wav2Vec2FeatureExtractor

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


class ASRInput(BaseModel):
    sound_wave: Any
    sampling_rate: Any


class ASROutput(BaseModel):
    embedding: Any


@Pipeline.register(task="asr", task_aliases=None, default_model_path=None)
class ASRPipeline(TransformersPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input_schema(self):
        return ASRInput

    @property
    def output_schema(self):
        return ASROutput

    def process_inputs(self, inputs):
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=inputs.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        features = feature_extractor(
            inputs.sound_wave, padding="max_length", max_length=2000, truncation=True, sampling_rate  = inputs.sampling_rate,

        ).input_values
        return features[0].reshape(1, 1, -1)

    def process_engine_outputs(self, x):
        return ASROutput(embedding=x[0].flatten())

    def route_input_to_bucket(self):
        pass
