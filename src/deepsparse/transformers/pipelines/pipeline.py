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

"""
Base Pipeline class for transformers inference pipeline
"""


from typing import Any, List, Mapping, Optional

import numpy
from transformers.models.auto import AutoConfig, AutoTokenizer

from deepsparse import DEEPSPARSE_ENGINE, Pipeline, Scheduler
from deepsparse.transformers.helpers import (
    get_onnx_path_and_configs,
    overwrite_transformer_onnx_model_inputs,
)


__all__ = ["TransformersPipeline"]


class TransformersPipeline(Pipeline):
    """
    Base deepsparse.Pipeline class for transformers model loading. This class handles
    the parsing of deepsparse-transformers files and model inputs, supporting loading
    from sparsezoo, a directory containing a model.onnx, tokenizer, and model config,
    or just an ONNX file with the ability to load a tokenizer and model config from
    a default huggingface-transformers model.

    Note, when implementing child tasks in deepsparse.transformers.pipelines,
    in addition to registering task names with Pipeline.register, task names should
    be added to the supported nlp tasks in deepsparse.tasks so they can be properly
    imported at runtime.

    :param model_path: sparsezoo stub to a transformers model, an ONNX file, or
        (preferred) a directory containing a model.onnx, tokenizer config, and model
        config. If no tokenizer and/or model config(s) are found, then they will be
        loaded from huggingface transformers using the `default_model_name` key
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: static batch size to use for inference. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param sequence_length: static sequence length to use for inference
    :param default_model_name: huggingface transformers model name to use to
        load a tokenizer and model config when none are provided in the `model_path`.
        Default is 'bert-base-uncased'
    """

    def __init__(
        self,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
        sequence_length: int = 128,
        default_model_name: str = "bert-base-uncased",
    ):

        self._sequence_length = sequence_length
        self._default_model_name = default_model_name

        self.config = None
        self.tokenizer = None
        self._onnx_input_names = None

        super().__init__(
            model_path=model_path,
            engine_type=engine_type,
            batch_size=batch_size,
            num_cores=num_cores,
            scheduler=scheduler,
            input_shapes=input_shapes,
            alias=alias,
        )

    @property
    def sequence_length(self) -> int:
        """
        :return: static sequence length to use for inference
        """
        return self._sequence_length

    @property
    def default_model_name(self) -> str:
        """
        :return: huggingface transformers model name to use to
            load a tokenizer and model config when none are provided in the
            `model_path`
        """
        return self._default_model_name

    def setup_onnx_file_path(self) -> str:
        """
        Parses ONNX, tokenizer, and config file paths from the given `model_path`.
        Supports sparsezoo stubs. If a tokenizer and/or config file are not found,
        they will be defaulted to the default_model_name in the transformers repo

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path, config_path, tokenizer_path = get_onnx_path_and_configs(
            self.model_path
        )

        # default config + tokenizer if necessary
        config_path = config_path or self.default_model_name
        tokenizer_path = tokenizer_path or self.default_model_name

        self.config = AutoConfig.from_pretrained(
            config_path, finetuning_task=self.task if hasattr(self, "task") else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, model_max_length=self.sequence_length
        )

        # overwrite onnx graph to given required input shape
        onnx_path, self._onnx_input_names, _ = overwrite_transformer_onnx_model_inputs(
            onnx_path, max_length=self.sequence_length
        )

        return onnx_path

    def tokens_to_engine_input(
        self, tokens: Mapping[Any, numpy.ndarray]
    ) -> List[numpy.ndarray]:
        """
        :param tokens: outputs of the pipeline tokenizer
        :return: list of numpy arrays in expected order for model input
        """
        if not all(name in tokens for name in self._onnx_input_names):
            raise ValueError(
                f"pipeline expected arrays with names {self._onnx_input_names}, "
                f"received inputs: {list(tokens.keys())}"
            )

        return [tokens[name] for name in self._onnx_input_names]
