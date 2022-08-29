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

import os
import warnings
from typing import Any, List, Mapping, Optional, Union

import numpy
from transformers.models.auto import AutoConfig, AutoTokenizer

from deepsparse import Bucketable, Pipeline
from deepsparse.transformers.helpers import (
    get_onnx_path_and_configs,
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
    """

    def __init__(
        self,
        *,
        sequence_length: Union[int, List[int]] = 128,
        **kwargs,
    ):

        self._sequence_length = sequence_length

        self.config = None
        self.tokenizer = None
        self.config_path = None
        self.tokenizer_config_path = None  # path to 'tokenizer.json'
        self.onnx_input_names = None

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
        Parses ONNX, tokenizer, and config file paths from the given `model_path`.
        Supports sparsezoo stubs

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path, config_path, tokenizer_path = get_onnx_path_and_configs(
            self.model_path, require_configs=True
        )

        self.config = AutoConfig.from_pretrained(
            config_path, finetuning_task=self.task if hasattr(self, "task") else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, model_max_length=self.sequence_length
        )
        self.config_path = os.path.join(config_path, "config.json")
        self.tokenizer_config_path = os.path.join(tokenizer_path, "tokenizer.json")

        # overwrite onnx graph to given required input shape
        (
            onnx_path,
            self.onnx_input_names,
            self._temp_model_directory,
        ) = overwrite_transformer_onnx_model_inputs(
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
        if not all(name in tokens for name in self.onnx_input_names):
            raise ValueError(
                f"pipeline expected arrays with names {self.onnx_input_names}, "
                f"received inputs: {list(tokens.keys())}"
            )

        return [tokens[name] for name in self.onnx_input_names]

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
