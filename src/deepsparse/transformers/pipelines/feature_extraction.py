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

# postprocessing adapted from huggingface/transformers

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Pipeline implementation and pydantic models for feature extraction transformers
tasks
"""


from typing import List, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "FeatureExtractionInput",
    "FeatureExtractionOutput",
    "FeatureExtractionPipeline",
]


class FeatureExtractionInput(BaseModel):
    """
    Schema for inputs to feature_extraction pipelines
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to"
        "feature_extraction task"
    )


class FeatureExtractionOutput(BaseModel):
    """
    Schema for feature_extraction pipeline output. Values are in batch order
    """

    embeddings: List[List[float]] = Field(
        description="The features computed by the model."
    )


@Pipeline.register(
    task="feature_extraction",
    task_aliases=[],
    default_model_path=(
        "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
        "wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni"
    ),
)
class FeatureExtractionPipeline(TransformersPipeline):
    """
    transformers feature extraction pipeline

    example instantiation:
    ```python
    feature_extractor = Pipeline.create(
        task="feature_extraction",
        model_path="feature_extraction_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```

    example batch size 1, single text inputs:
    ```python
    sentiment = feature_extractor("the food tastes great")
    sentiment = feature_extractor(["the food tastes great"])
    sentiment = feature_extractor([["the food tastes great"]])
    ```

    example batch size 1, multi text input:
    ```python
    prediction = feature_extractor([["how is the food?", "what is the food?"]])
    ```

    example batch size n, single text inputs:
    ```python
    sentiments = feature_extractor(["the food tastes great", "the food tastes bad"])
    sentiments = feature_extractor([["the food tastes great"], ["the food tastes bad"]])
    ```

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
    :param sequence_length: sequence length to compile model and tokenizer for.
        Default is 128
    :param default_model_name: huggingface transformers model name to use to
        load a tokenizer and model config when none are provided in the `model_path`.
        Default is 'bert-base-uncased'
    :param return_all_scores: if True, instead of returning the prediction as the
        argmax of model class predictions, will return all scores and labels as
        a list for each result in the batch. Default is False
    """

    def __init__(
        self,
        *,
        return_all_scores: bool = False,
        **kwargs,
    ):
        self._return_all_scores = return_all_scores

        super().__init__(**kwargs)

    @property
    def return_all_scores(self) -> str:
        """
        :return: if True, instead of returning the prediction as the
            argmax of model class predictions, will return all scores and labels as
            a list for each result in the batch
        """
        return self._return_all_scores

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return FeatureExtractionInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return FeatureExtractionOutput

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
            if len(args) == 1:
                # passed input_schema schema directly
                if isinstance(args[0], self.input_schema):
                    return args[0]
                return self.input_schema(sequences=args[0])
            else:
                return self.input_schema(sequences=args)

        return self.input_schema(**kwargs)

    def process_inputs(self, inputs: FeatureExtractionInput) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            FeatureExtractionInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        tokens = self.tokenizer(
            inputs.sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
        )
        return self.tokens_to_engine_input(tokens)

    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray]) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        outputs = engine_outputs
        if isinstance(outputs, list):
            outputs = outputs[0]

        scores = (
            1.0 / (1.0 + numpy.exp(-outputs))
            if self.config.num_labels == 1
            else numpy.exp(outputs) / numpy.exp(outputs).sum(-1, keepdims=True)
        )

        if not self._return_all_scores:
            # return only argmax of scores for each item in batch
            labels = []
            label_scores = []
            for score in scores:
                labels.append(self.config.id2label[score.argmax()])
                label_scores.append(score.max().item())
        else:
            # return all scores and labels for each item in batch
            labels = [
                [self.config.id2label[idx] for idx in range(scores.shape[1])]
            ] * len(scores)
            label_scores = [score.reshape(-1).tolist() for score in scores]

        return self.output_schema(
            labels=labels,
            scores=label_scores,
        )
