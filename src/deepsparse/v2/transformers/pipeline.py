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
from typing import Optional

import transformers
from transformers import AutoTokenizer

from deepsparse.transformers.helpers import get_deployment_path
from deepsparse.v2.pipeline import Pipeline


__all__ = ["TransformersPipeline"]


class TransformersPipeline(Pipeline):
    def __init__(self, **kwargs):
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        super().__init__(**kwargs)

    def setup_onnx_file_path(
        self,
        model_path: str,
        sequence_length: int,
        onnx_model_name: Optional[str] = None,
    ) -> str:
        """
        Parses ONNX model from the `model_path` provided. It additionally
        creates config and tokenizer objects from the `deployment path`,
        derived from the `model_path` provided.

        :param model_path: path to the model to be parsed
        :param sequence_length: maximum sequence length of the model
        :param onnx_model_name: optionally, the precise name of the ONNX model
            of interest may be specified. If not specified, the default ONNX model
            name will be used (refer to `get_deployment_path` for details)
        :return: file path to the processed ONNX file for the engine to compile
        """
        deployment_path, onnx_path = get_deployment_path(model_path, onnx_model_name)

        hf_logger = logging.getLogger("transformers")
        hf_logger_level = hf_logger.level
        hf_logger.setLevel(logging.ERROR)

        self.config = transformers.PretrainedConfig.from_pretrained(
            deployment_path,
            finetuning_task=self.task if hasattr(self, "task") else None,
        )
        hf_logger.setLevel(hf_logger_level)

        self._trust_remote_code = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            deployment_path,
            trust_remote_code=self._trust_remote_code,
            model_max_length=sequence_length,
        )

        if not self.config or not self.tokenizer:
            raise RuntimeError(
                "Invalid config or tokenizer provided. Please provide "
                "paths to the files or ensure they exist in the `model_path` provided. "
                "See `tokenizer` and `config` arguments for details."
            )
        return onnx_path
