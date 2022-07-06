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

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Optional, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.engine import Context
from deepsparse.transformers.pipelines import TransformersPipeline


class ZeroShotTextClassificationOutput(BaseModel):
    """
    Schema for zero_shot_text_classification pipeline output. Values are in batch order
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to "
        "zero_shot_text_classification task"
    )
    labels: Union[List[List[str]], List[str]] = Field(
        description="The predicted labels in batch order"
    )
    scores: Union[List[List[float]], List[float]] = Field(
        description="The corresponding probability for each label in the batch"
    )


class ZeroShotTextClassificationImplementation(TransformersPipeline):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _parse_config(
        self, model_scheme_config: Optional[Union[BaseModel, dict]]
    ) -> Type[BaseModel]:
        """
        :param model_scheme_config: optional config arguments specified by user
        :return: instance of config pydantic model for this pipeline's model scheme
        """
        model_scheme_config = model_scheme_config if model_scheme_config else {}

        if isinstance(model_scheme_config, self.config_schema):
            return model_scheme_config

        elif isinstance(model_scheme_config, dict):
            return self.config_schema(**model_scheme_config)

        else:
            raise ValueError(
                f"pipeline {self.__class__} only supports either only a "
                f"{config_schema} object a dict of keywords used to construct "
                f"one. Found {model_scheme_config} instead"
            )

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return ZeroShotTextClassificationOutput

    def _parse_labels(self, labels: Union[None, List[str], str]) -> List[str]:
        """
        If given a string of comma separated labels, parses values into a list

        :param labels: A string of comma separated labels or a list of labels
        :return: a list of labels, parsed if originally in string form
        """
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",") if label.strip()]
        return labels

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        """
        :param args: ordered arguments to pipeline, only an input_schema object
            is supported as an arg for this function
        :param kwargs: keyword arguments to pipeline
        :return: pipeline arguments parsed into the given `input_schema`
            schema if necessary. If an instance of the `input_schema` is provided
            it will be returned
        """
        if args and kwargs:
            raise ValueError(
                f"{self.__class__} only support args OR kwargs. Found "
                f" {len(args)} args and {len(kwargs)} kwargs"
            )

        if args:
            if len(args) == 1 and isinstance(args[0], self.input_schema):
                input = args[0]
            else:
                input = self.input_schema(*args)
        else:
            input = self.input_schema(**kwargs)

        return input

    @staticmethod
    def route_input_to_bucket(
        cls, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        current_seq_len = cls.get_current_sequence_length(input_schema)

        for pipeline in pipelines:
            if pipeline.sequence_length > current_seq_len:
                return pipeline
        return pipelines[-1]

    @staticmethod
    @abstractmethod
    def get_current_sequence_length(input_schema: BaseModel) -> int:
        """
        Helper function to get max sequence length in provided sequences input

        :param input_schema: input to pipeline
        :return: max sequence length in input_schema
        """
        raise NotImplementedError()
