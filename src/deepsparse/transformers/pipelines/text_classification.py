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
Pipeline implementation and pydantic models for text classification transformers
tasks
"""


import warnings
from typing import List, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "TextClassificationInput",
    "TextClassificationOutput",
    "TextClassificationPipeline",
]


class TextClassificationInput(BaseModel):
    """
    Schema for inputs to text_classification pipelines
    """

    sequences: Union[List[List[str]], List[str], str] = Field(
        description="A string or List of strings representing input to"
        "text_classification task"
    )


class TextClassificationOutput(BaseModel):
    """
    Schema for text_classification pipeline output. Values are in batch order
    """

    labels: List[Union[str, List[str]]] = Field(
        description="The predicted labels in batch order"
    )
    scores: List[Union[float, List[float]]] = Field(
        description="The corresponding probability for each label in the batch"
    )


@Pipeline.register(
    task="text_classification",
    task_aliases=["glue", "sentiment_analysis"],
    default_model_path=(
        "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/"
        "sst2/12layer_pruned80_quant-none-vnni"
    ),
)
class TextClassificationPipeline(TransformersPipeline):
    """
    transformers text classification pipeline.
    Scores are returned as sigmoid over logits if one label is given or
    the model config is set to "multi_label_classification". Softmax
    returned otherwise

    example instantiation:
    ```python
    text_classifier = Pipeline.create(
        task="text_classification",
        model_path="text_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```

    example batch size 1, single text inputs (ie sentiment analysis):
    ```python
    sentiment = text_classifier("the food tastes great")
    sentiment = text_classifier(["the food tastes great"])
    sentiment = text_classifier([["the food tastes great"]])
    ```

    example batch size 1, multi text input (ie QQP like tasks):
    ```python
    prediction = text_classifier([["how is the food?", "what is the food?"]])
    ```

    example batch size n, single text inputs:
    ```python
    sentiments = text_classifier(["the food tastes great", "the food tastes bad"])
    sentiments = text_classifier([["the food tastes great"], ["the food tastes bad"]])
    ```

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
    :param top_k: number of labels with the highest score to return. Default is 1
    :param return_all_scores: [DEPRECATED] set top_k to desired value instead.
        If True, instead of returning the prediction as the
        argmax of model class predictions, will return all scores and labels as
        a list for each result in the batch. Default is False
    """

    def __init__(
        self,
        *,
        top_k: int = 1,
        return_all_scores: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._top_k = _get_top_k(top_k, return_all_scores, self.config.num_labels)
        self._return_all_scores = return_all_scores

    @property
    def top_k(self) -> int:
        """
        :return: number of labels with the highest score to return
        """
        return self._top_k

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
        return TextClassificationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return TextClassificationOutput

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

    def process_inputs(self, inputs: TextClassificationInput) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            TextClassificationInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        sequences = inputs.sequences
        if isinstance(sequences, List) and all(
            isinstance(sequence, List) and len(sequence) == 1 for sequence in sequences
        ):
            # if batch items contain only one sequence but are wrapped in lists, unwrap
            # for use as tokenizer input
            sequences = [sequence[0] for sequence in sequences]

        tokens = self.tokenizer(
            sequences,
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

        use_sigmoid = self.config.num_labels == 1 or (
            self.config.problem_type == "multi_label_classification"
        )
        scores = (
            1.0 / (1.0 + numpy.exp(-outputs))
            if use_sigmoid
            else numpy.exp(outputs) / numpy.exp(outputs).sum(-1, keepdims=True)
        )

        labels = []
        label_scores = []
        for score in scores:
            if self._top_k == 1:
                labels.append(self.config.id2label[score.argmax()])
                label_scores.append(score.max().item())
            else:
                ranked_idxs = (-score.reshape(-1)).argsort()[: self.top_k]
                score = score.reshape(-1).tolist()
                labels.append([self.config.id2label[idx] for idx in ranked_idxs])
                label_scores.append([score[idx] for idx in ranked_idxs])

        return self.output_schema(
            labels=labels,
            scores=label_scores,
        )

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[TransformersPipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        tokenizer = pipelines[0].tokenizer
        tokens = tokenizer(
            input_schema.sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=False,
            truncation=False,
        )
        input_seq_len = max(map(len, tokens["input_ids"]))
        return TransformersPipeline.select_bucket_by_seq_len(input_seq_len, pipelines)


def _get_top_k(top_k: int, return_all_scores: bool, num_labels: int) -> int:
    # handle deprecations of return_all_scores
    if top_k <= 0:
        raise ValueError(f"text_classification top_k must be > 0. found {top_k}")
    if return_all_scores and top_k not in [1, num_labels]:
        # top_k is not set to default, but does not match num_labels
        # for return_all_scores
        raise ValueError(
            f"Mismatch: return_all_scores set to {return_all_scores} with "
            f"{num_labels} in model config, but top_k set to {top_k}. "
            "return_all_scores is deprecated, set top_k instead"
        )

    if top_k == 1 and return_all_scores:
        # set top_k to num_labels from config and warn
        warnings.warn(
            f"return_all_scores deprecated, set {top_k} instead. setting "
            f"top_k to num_lables from config: {num_labels}",
            category=DeprecationWarning,
        )
        top_k = num_labels

    return top_k
