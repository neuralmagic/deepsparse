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

from deepsparse.v2.text_generation import KVCacheCreator, KVCacheCreatorInput


def test_kv_cache_creation(
    text_generation_attributes, model_attributes, pipeline_state
):
    """
    Check if the KVCacheCreator successfully creates a kv_cache object, given the
    single_token_engine attributes stored in the pipeline_state.
    """
    seq_length, prompt_seq_len = text_generation_attributes
    tokenizer, _ = model_attributes
    kv_cache_creator = KVCacheCreator(
        tokenizer=tokenizer,
        prompt_sequence_length=prompt_seq_len,
        sequence_length=seq_length,
        internal_kv_cache=False,
    )

    assert kv_cache_creator.input_schema == KVCacheCreatorInput
    kv_cache = kv_cache_creator.run(
        cache_shape=pipeline_state.current_state.get("cache_shape"),
        kv_cache_data_type=pipeline_state.current_state.get("kv_cache_data_type"),
        output_names=pipeline_state.current_state.get("output_names"),
    )
    assert kv_cache.get("kv_cache")
    assert kv_cache.get("kv_cache").total_num_processed_tokens == 0
