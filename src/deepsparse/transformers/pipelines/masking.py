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
Pipeline implementation and pydantic models for masking transformers
tasks
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.file_utils import ExplicitEnum
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline

__all__ = [
    "MaskingInput",
    "MaskingResult",
    "MaskingOutput",
    "MaskingPipeline",
]


class MaskingInput(BaseModel):
    """
    Schema for inputs to masking pipelines
    """

    inputs: Union[List[str], str] = Field(
        description=(
            "A string or List of batch of strings representing input(s) to"
            "a masking task"
        )
    )
    is_split_into_words: bool = Field(
        default=False,
        description=(
            "True if the input is a batch size 1 list of strings representing. "
            "individual word tokens. Currently only supports batch size 1. "
            "Default is False"
        ),
    )


class MaskingResult(BaseModel):
    """
    Schema for a masking of a single token
    """

    sentence_id: int = Field(description="sentence id")
    mask_token_id: int = Field(description="token id")
    topK_tokens: List[str] = Field(description="TopK tokens which can be replaced with mask")
    topK_token_ids: List[int] = Field(description="TopK token ids which can be replaced with mask")
    topK_logits: List[float] = Field(description="TopK Logits values")


class MaskingOutput(BaseModel):
    """
    Schema for results of MaskingPipeline inference. Classifications of each
    token stored in a list of lists of batch[sentence[token]]
    """

    predictions: List[List[MaskingResult]] = Field(
        description=(
            "list of list of results of token classification pipeline. Outer list "
            "has one item for each sequence in the batch. Inner list has one "
            "TokenClassificationResult item per token in the given sequence"
        )
    )


@Pipeline.register(
    task="masking",
    task_aliases=["fill-in-the-blank"],
    default_model_path=(
            "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/"
            "12layer_pruned80_quant-none-vnni"
    ),
)
class MaskingPipeline(TransformersPipeline):
    """
    This pipeline will enable to use to [MASK] tokens and generate relevant information from it.

    example instantiation:
    ```python
    masking = Pipeline.create(
        task="masking",
        model_path="mlm_model_path/",
        batch_size=BATCH_SIZE,
    )
    ```
    Example code:
    ```python
    from deepsparse.transformers import pipeline
    masking_pipeline = pipeline(
        task="masking",
        model_path= "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni",
        batch_size=2,
        top_k=2
    )
    token_classes = masking_pipeline(["I am [MASK] boy", "I am [MASK] guitar at [MASK]."])
    print(token_classes)
    ```
    Output:
    [[MaskingResult(sentence_id=0, mask_token_id=3, topK_tokens=['a', 'my'], topK_token_ids=[1037, 2026],
    topK_logits=[9.532556533813477, 9.05823802947998])], [MaskingResult(sentence_id=1, mask_token_id=3,
    topK_tokens=['playing', 'bass'], topK_token_ids=[2652, 3321], topK_logits=[9.855772972106934, 9.259352684020996]),
    MaskingResult(sentence_id=1, mask_token_id=6, topK_tokens=['home', 'sea'], topK_token_ids=[2188, 2712],
    topK_logits=[6.91043758392334, 6.050468921661377])]]


    The output will be returning list of list based on the batch size. The output will be storing list of
    MaskingResult object.
    MaskingResult will be returning:
        - sentence_id: sentence id
        - mask_token_id: position of mask token
        - topK_tokens: List of tokens that can be replaced with MASK token for that position.
        - topK_token_ids: List of token ids
        - topK_logits: List of logit values representing to tokens


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
    :param top_k: Number of tokens to retrieve for a given mask position

    """

    def __init__(
            self,
            *,
            top_k: int = 3,
            **kwargs,
    ):

        self.top_k = top_k
        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return MaskingInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return MaskingOutput

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
                return self.input_schema(inputs=args[0])
            else:
                return self.input_schema(inputs=args)

        return self.input_schema(**kwargs)

    def process_inputs(
            self,
            inputs: MaskingInput,
    ) -> Tuple[List[numpy.ndarray], Dict[str, Any]]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            MaskingInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
            and dictionary containing offset mappings and special tokens mask to
            be used during postprocessing
        """
        if inputs.is_split_into_words and self.engine.batch_size != 1:
            raise ValueError("is_split_into_words=True only supported for batch size 1")

        tokens = self.tokenizer(
            inputs.inputs,
            return_tensors="np",
            truncation=TruncationStrategy.LONGEST_FIRST.value,
            padding=PaddingStrategy.MAX_LENGTH.value,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            is_split_into_words=inputs.is_split_into_words,
        )

        offset_mapping = (
            tokens.pop("offset_mapping")
            if self.tokenizer.is_fast
            else [None] * len(inputs.inputs)
        )
        special_tokens_mask = tokens.pop("special_tokens_mask")

        word_start_mask = None
        if inputs.is_split_into_words:
            # create mask for word in the split words where values are True
            # if they are the start of a tokenized word
            word_start_mask = []
            word_ids = tokens.word_ids(batch_index=0)
            previous_id = None
            for word_id in word_ids:
                if word_id is None:
                    continue
                if word_id != previous_id:
                    word_start_mask.append(True)
                    previous_id = word_id
                else:
                    word_start_mask.append(False)

        postprocessing_kwargs = dict(
            inputs=inputs,
            tokens=tokens,
            offset_mapping=offset_mapping,
            special_tokens_mask=special_tokens_mask,
            word_start_mask=word_start_mask,
        )

        return self.tokens_to_engine_input(tokens), postprocessing_kwargs

    def process_engine_outputs(
            self,
            engine_outputs: List[numpy.ndarray],
            **kwargs,
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """

        tokens = kwargs["tokens"]
        token_ids = tokens["input_ids"]
        masked_positions = (token_ids == self.tokenizer.mask_token_id).nonzero()
        predictions = []  # type: List[List[MaskingResult]]

        sentence_ids, mask_position = masked_positions
        for index, sentence_index in enumerate(sentence_ids):

            sentence_output = engine_outputs[0][sentence_index]
            masked_values = sentence_output[mask_position[index]]
            sorted_token_ids = numpy.argsort(masked_values)[::-1]
            topK_token_ids = list(sorted_token_ids[:self.top_k])
            topK_logits = list(masked_values[topK_token_ids])
            topK_tokens = self.tokenizer.batch_decode(topK_token_ids)
            output = MaskingResult(sentence_id=sentence_index,
                                   mask_token_id=mask_position[index],
                                   topK_tokens=topK_tokens,
                                   topK_logits=topK_logits,
                                   topK_token_ids=topK_token_ids

                                   )
            if sentence_index <= len(predictions) - 1:
                predictions[sentence_index].append(output)
            else:
                predictions.append([output])
        return self.output_schema(predictions=predictions)

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
            input_schema.inputs,
            add_special_tokens=True,
            return_tensors="np",
            padding=False,
            truncation=False,
        )
        input_seq_len = max(map(len, tokens["input_ids"]))
        return TransformersPipeline.select_bucket_by_seq_len(input_seq_len, pipelines)
