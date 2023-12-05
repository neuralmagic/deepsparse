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

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy
import transformers
from transformers.models.auto import AutoTokenizer

from deepsparse import Bucketable, Pipeline
from deepsparse.transformers.helpers import (
    get_deployment_path,
    overwrite_transformer_onnx_model_inputs,
)


__all__ = [
    "TransformersPipeline",
    "pipeline",
]


class TransformersPipeline(Pipeline, Bucketable):
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

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param trust_remote_code: if True, will trust remote code. This option
        should only be set to `True` for repositories you trust and in which
        you have read the code, as it will execute possibly unsafe code
        on your local machine. Default is False
    :param config: hugging face transformers model config. Can be a path to the config,
        a dictionary with the config values, a transformers.PretrainedConfig, or None.
        If a directory is provided, it is assumed that the file is named config.json.
        If None, an attempt is made to read the config file from the the model_path
        directory provided. Default is None
    :param tokenizer: hugging face transfromers tokenizer. Can be a path to the
        a directory with the relevant tokenizer files, a
        transformers.PreTrainedTokenizerBase, or None. If None, an attempt is made to
        read the json file from the model_path directory provided. Default is None.
    """

    def __init__(
        self,
        *,
        sequence_length: Union[int, List[int]] = 128,
        trust_remote_code: bool = False,
        config: Union[str, Path, Dict, transformers.PretrainedConfig] = None,
        tokenizer: Union[str, Path, transformers.PreTrainedTokenizerBase] = None,
        **kwargs,
    ):

        self._sequence_length = sequence_length
        self._trust_remote_code = trust_remote_code

        self.config = config
        self.tokenizer = tokenizer

        self._deployment_path = None
        self.onnx_input_names = None
        self._delay_overwriting_inputs = (
            kwargs.pop("_delay_overwriting_inputs", None) or False
        )
        self._temp_model_directory = None

        super().__init__(**kwargs)

    @property
    def sequence_length(self) -> Union[int, List[int]]:
        """
        :return: static sequence length to use for inference
        """
        return self._sequence_length

    def setup_onnx_file_path(self) -> str:
        """
        Parses ONNX model from the `model_path` provided. It additionally
        creates config and tokenizer objects from the `deployment path`,
        derived from the `model_path` provided.

        :return: file path to the processed ONNX file for the engine to compile
        """

        deployment_path, onnx_path = get_deployment_path(self.model_path)
        self._deployment_path = deployment_path

        # temporarily set transformers logger to ERROR to avoid
        # printing misleading warnings
        hf_logger = logging.getLogger("transformers")
        hf_logger_level = hf_logger.level
        hf_logger.setLevel(logging.ERROR)
        self.config = transformers.PretrainedConfig.from_pretrained(
            deployment_path,
            finetuning_task=self.task if hasattr(self, "task") else None,
        )
        hf_logger.setLevel(hf_logger_level)

        self.tokenizer = AutoTokenizer.from_pretrained(
            deployment_path,
            trust_remote_code=self._trust_remote_code,
            model_max_length=self.sequence_length,
        )

        if not self._delay_overwriting_inputs:
            # overwrite onnx graph to given required input shape
            (
                onnx_path,
                self.onnx_input_names,
                self._temp_model_directory,
            ) = overwrite_transformer_onnx_model_inputs(
                onnx_path, max_length=self.sequence_length
            )

        if not self.config or not self.tokenizer:
            raise RuntimeError(
                "Invalid config or tokenizer provided. Please provide "
                "paths to the files or ensure they exist in the `model_path` provided. "
                "See `tokenizer` and `config` arguments for details."
            )
        return onnx_path

    def tokens_to_engine_input(
        self,
        tokens: Mapping[Any, numpy.ndarray],
        onnx_input_names: Optional[List[str]] = None,
    ) -> List[numpy.ndarray]:
        """
        :param tokens: outputs of the pipeline tokenizer
        :return: list of numpy arrays in expected order for model input
        """
        if onnx_input_names is None:
            onnx_input_names = self.onnx_input_names
        if not all(name in tokens for name in onnx_input_names):
            raise ValueError(
                f"pipeline expected arrays with names {onnx_input_names}, "
                f"received inputs: {list(tokens.keys())}"
            )

        return [tokens[name] for name in onnx_input_names]

    @staticmethod
    def should_bucket(*args, **kwargs) -> bool:
        """
        :returns: True if kwargs contain sequence_length as a list; otherwise False
        """
        sequence_length = kwargs.get("sequence_length", 128)
        return isinstance(sequence_length, list)

    @staticmethod
    def create_pipeline_buckets(
        *args, sequence_length: List[int], **kwargs
    ) -> List[Pipeline]:
        """
        Create and return a list of Pipeline objects representing different
        buckets

        :param args: args for pipeline creation
        :param sequence_length: a List of sequence lengths to initialize buckets
            for
        :param kwargs: keyword args for pipeline creation
        :return: A List[Pipeline] objects representing different buckets
        """
        pipelines = []
        for seq_len in sorted(sequence_length):
            curr_pipeline = Pipeline.create(*args, sequence_length=seq_len, **kwargs)
            pipelines.append(curr_pipeline)

        return pipelines

    @staticmethod
    def select_bucket_by_seq_len(
        input_seq_len: int, buckets: List["TransformersPipeline"]
    ) -> "TransformersPipeline":
        """
        :param input_seq_len: sequence length to select a bucket for
        :param buckets: A List of Pipeline objects representing different buckets
        :return: pipeline with the minimal sequence length to fit the input sequence
            length. If no pipeline fits the input, the pipeline with the largest
            sequence length is returned
        """
        valid_pipelines = [
            bucket for bucket in buckets if bucket.sequence_length >= input_seq_len
        ]
        if len(valid_pipelines) == 0:
            return max(buckets, key=lambda bucket: bucket.sequence_length)
        return min(valid_pipelines, key=lambda bucket: bucket.sequence_length)

    @property
    def config_path(self) -> str:
        """
        :return: full path to config.json for this pipeline if it exists,
            otherwise returns deployment directory path
        """
        config_path = os.path.join(self._deployment_path, "config.json")
        if os.path.exists(config_path):
            return config_path
        else:
            return self._deployment_path

    @property
    def tokenizer_config_path(self) -> str:
        """
        :return: full path to tokenizer.json for this pipeline if it exists,
            otherwise returns deployment directory path
        """
        tokenizer_path = os.path.join(self._deployment_path, "tokenizer.json")
        tokenizer_config_path = os.path.join(
            self._deployment_path, "tokenizer_config.json"
        )
        if os.path.exists(tokenizer_path):
            return tokenizer_path
        elif os.path.exists(tokenizer_config_path):
            return tokenizer_path
        return self._deployment_path


def pipeline(
    task: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    engine_type: str = "deepsparse",
    config: Optional[str] = None,
    tokenizer: Optional[str] = None,
    max_length: int = 128,
    num_cores: Optional[int] = None,
    scheduler: Optional[str] = None,
    batch_size: Optional[int] = 1,
    **kwargs,
) -> Pipeline:
    """
    [DEPRECATED] - deepsparse.transformers.pipeline is deprecated to create DeepSparse
    pipelines for transformers tasks use deepsparse.Pipeline.create(task, ...)

    Utility factory method to build a Pipeline

    :param task: name of the task to define which pipeline to create. Currently,
        supported task - "question-answering"
    :param model_name: canonical name of the hugging face model this model is based on
    :param model_path: path to model directory containing `model.onnx`, `config.json`,
        and `tokenizer.json` files, ONNX model file, or SparseZoo stub
    :param engine_type: inference engine name to use. Options are 'deepsparse'
        and 'onnxruntime'. Default is 'deepsparse'
    :param config: huggingface model config, if none provided, default will be used
        which will be from the model name or sparsezoo stub if given for model path
    :param tokenizer: huggingface tokenizer, if none provided, default will be used
    :param max_length: maximum sequence length of model inputs. default is 128
    :param num_cores: number of CPU cores to run engine with. Default is the maximum
        available
    :param scheduler: The scheduler to use for the engine. Can be None, single or multi
    :param batch_size: The batch_size to use for the pipeline. Defaults to 1
        Note: `question-answering` pipeline only supports a batch_size of 1.
    :param kwargs: additional key word arguments for task specific pipeline constructor
    :return: Pipeline object for the given taks and model
    """
    warnings.warn(
        "[DEPRECATED] - deepsparse.transformers.pipeline is deprecated to create "
        "DeepSparse pipelines for transformers tasks use deepsparse.Pipeline.create()"
    )

    if config is not None or tokenizer is not None:
        raise ValueError(
            "Directly passing in a config or tokenizer to DeepSparse transformers "
            "pipelines is no longer supported. config and tokenizer objects should "
            "be specified by including config.json and tokenizer.json files in the "
            "model directory respectively"
        )

    return Pipeline.create(
        task=task,
        model_path=model_path,
        engine_type=engine_type,
        batch_size=batch_size,
        num_cores=num_cores,
        scheduler=scheduler,
        sequence_length=max_length,
        **kwargs,
    )
