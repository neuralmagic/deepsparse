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

from typing import Any, Dict, Optional, Tuple

import transformers

from deepsparse.transformers.helpers import setup_onnx_file_path


__all__ = ["setup_transformers_pipeline"]


def setup_transformers_pipeline(
    model_path: str,
    sequence_length: int,
    tokenizer_padding_side: str = "left",
    engine_kwargs: Optional[Dict] = None,
    onnx_model_name: Optional[str] = None,
) -> Tuple[
    str, transformers.PretrainedConfig, transformers.PreTrainedTokenizer, Dict[str, Any]
]:
    """
    A helper function that sets up the model path, config, tokenizer,
    and engine kwargs for a transformers model.

    :param model_path: The path to the model to load
    :param sequence_length: The sequence length to use for the model
    :param tokenizer_padding_side: The side to pad on for the tokenizer,
        either "left" or "right"
    :param engine_kwargs: The kwargs to pass to the engine
    :param onnx_model_name: The name of the onnx model to be loaded.
        If not specified, defaults are used (see setup_onnx_file_path)
    :return The model path, config, tokenizer, and engine kwargs
    """
    model_path, config, tokenizer = setup_onnx_file_path(
        model_path, sequence_length, onnx_model_name
    )

    tokenizer.padding_side = tokenizer_padding_side
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    engine_kwargs = engine_kwargs or {}
    engine_kwargs["model_path"] = model_path
    return model_path, config, tokenizer, engine_kwargs
