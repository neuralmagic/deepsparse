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
from typing import Any, Dict, List, Optional

import numpy
import onnx

from deepsparse import Pipeline
from deepsparse.engine import Context, Engine
from deepsparse.transformers.utils.helpers import softmax
from sparsezoo.utils.onnx import save_onnx


_LOGGER = logging.getLogger(__name__)

__all__ = ["NLDecoderEngine"]

_CACHE_INPUT_NAME = "past_key_values"


class NLDecoderEngine(Engine):
    def __init__(
        self,
        onnx_file_path: str,
        engine_type: str,
        multitoken: bool,
        engine_args: Dict[str, Any],
        sequence_length: int,  # a bit ugly
        engine_context: Optional[Context] = None,
        sampling_temperature: float = 1.0,
        deterministic: bool = True,
    ):

        onnx_file_path, kv_cache_enabled = self.overwrite_onnx_model_inputs(
            onnx_file_path=onnx_file_path,
            batch_size=engine_args.get("batch_size", 1),
            sequence_length=sequence_length,
            multitoken=multitoken,
        )
        self.engine = Pipeline.create_engine(
            onnx_file_path=onnx_file_path,
            engine_type=engine_type,
            engine_args=engine_args,
            context=engine_context,
        )
        self.multitoken = multitoken
        self.sampling_temperature = sampling_temperature
        self.deterministic = deterministic
        self.kv_cache_enabled = kv_cache_enabled
        self.kv_cache = DecoderKVCache(engine_type) if kv_cache_enabled else None

    def __call__(
        self, inp: List[numpy.ndarray], val_inp: bool = True
    ) -> List[numpy.ndarray]:
        if self.is_cache_enabled:
            pass
        out = self.engine.run(inp, val_inp)
        return out

    @property
    def onnx_input_names_no_cache(self):
        return [
            name
            for name in self.engine.input_names
            if not name.startswith(_CACHE_INPUT_NAME)
        ]

    def overwrite_onnx_model_inputs(
        self,
        onnx_file_path: str,
        batch_size: int = 1,
        sequence_length: int = 128,
        multitoken: bool = False,
    ) -> str:

        model = onnx.load(onnx_file_path, load_external_data=False)
        initializer_input_names = set([node.name for node in model.graph.initializer])
        is_cache_enabled = any(
            _CACHE_INPUT_NAME in node.name for node in model.graph.initializer
        )
        external_inputs = [
            inp for inp in model.graph.input if inp.name not in initializer_input_names
        ]
        input_names = []

        for external_input in external_inputs:
            external_input.type.tensor_type.shape.dim[0].dim_value = batch_size

            if external_input.name in ["input_ids", "positions"]:
                external_input.type.tensor_type.shape.dim[1].dim_value = (
                    sequence_length if multitoken else 1
                )
            elif external_input.name == "attention_mask":
                # regardless of multi-token or not, always provide full attention mask
                external_input.type.tensor_type.shape.dim[1].dim_value = sequence_length

            elif external_input.name.startswith(_CACHE_INPUT_NAME):
                # empty cache for multi-token runs,
                # otherwise max cache len is max len - 1
                external_input.type.tensor_type.shape.dim[2].dim_value = (
                    0 if multitoken else sequence_length - 1
                )
            else:
                raise ValueError(
                    f"Unexpected external input name: {external_input.name}"
                )
            input_names.append(external_input.name)

        _LOGGER.info(
            f"Overwriting in-place the input shapes of the transformer model at {onnx_file_path}"
        )
        save_onnx(model, onnx_file_path)
        return onnx_file_path, is_cache_enabled

    def generate_token(self, logits: numpy.ndarray) -> numpy.ndarray:
        """
        Samples a token from the logits using the sampling temperature.
        :param logits: the logits from the model with shape (vocab_size,)
        :return: the sampled token
        """
        if self.deterministic:
            return numpy.argmax(logits)

        logits /= self.sampling_temperature

        probs = softmax(logits)

        return numpy.random.choice(len(probs), p=probs)
