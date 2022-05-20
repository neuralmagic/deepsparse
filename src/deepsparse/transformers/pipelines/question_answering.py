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
Pipeline implementation and pydantic models for question answering transformers
tasks
"""


from typing import Any, Dict, List, Tuple, Type

import numpy
from pydantic import BaseModel, Field
from transformers.data import (
    SquadExample,
    SquadFeatures,
    squad_convert_examples_to_features,
)
from transformers.tokenization_utils_base import PaddingStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "QuestionAnsweringInput",
    "QuestionAnsweringOutput",
    "QuestionAnsweringPipeline",
]


class QuestionAnsweringInput(BaseModel):
    """
    Schema for inputs to question_answering pipelines
    """

    question: str = Field(description="String question to be answered")
    context: str = Field(description="String representing context for answer")


class QuestionAnsweringOutput(BaseModel):
    """
    Schema for question_answering pipeline output. Values are in batch order
    """

    score: float = Field(description="confidence score for prediction")
    answer: str = Field(description="predicted answer")
    start: int = Field(description="start index of the answer")
    end: int = Field(description="end index of the answer")


@Pipeline.register(
    task="question_answering",
    task_aliases=["qa"],
    default_model_path=(
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/"
        "squad/12layer_pruned80_quant-none-vnni"
    ),
)
class QuestionAnsweringPipeline(TransformersPipeline):
    """
    transformers question_answering pipeline

    example instantiation:
    ```python
    question_answering = Pipeline.create(
        task="question_answering",
        model_path="question_answering_model_dir/",
    )
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
    :param doc_stride: if the context is too long to fit with the question for the
        model, it will be split in several chunks with some overlap. This argument
        controls the size of that overlap. Currently, only reading the first span
        is supported (everything after doc_stride will be truncated). Default
        is 128
    :param max_question_len: maximum length of the question after tokenization.
        It will be truncated if needed. Default is 64
    :param max_answer_len: maximum length of answer after decoding. Default is 15
    """

    def __init__(
        self,
        *,
        doc_stride: int = 128,
        max_question_length: int = 64,
        max_answer_length: int = 15,
        **kwargs,
    ):

        if kwargs.get("batch_size") and kwargs["batch_size"] > 1:
            raise ValueError(
                f"{self.__class__.__name__} currently only supports batch size 1, "
                f"batch size set to {kwargs['batch_size']}"
            )

        self._doc_stride = doc_stride
        self._max_question_length = max_question_length
        self._max_answer_length = max_answer_length

        super().__init__(**kwargs)

    @property
    def doc_stride(self) -> int:
        """
        :return: if the context is too long to fit with the question for the
            model, it will be split in several chunks with some overlap. This argument
            controls the size of that overlap. Currently, only reading the first span
            is supported (everything after doc_stride will be truncated)
        """
        return self._doc_stride

    @property
    def max_answer_length(self) -> int:
        """
        :return: maximum length of answer after decoding
        """
        return self._max_answer_length

    @property
    def max_question_length(self) -> int:
        """
        :return: maximum length of the question after tokenization.
            It will be truncated if needed
        """
        return self._max_question_length

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return QuestionAnsweringInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return QuestionAnsweringOutput

    def engine_forward(
        self,
        engine_inputs: List[List[numpy.ndarray]],
    ) -> List[List[numpy.ndarray]]:
        """
        runs one forward pass for each span of the preprocessed input

        :param engine_inputs: list of multiple inputs to engine forward pass
        :return: result of each forward pass
        """
        engine_outputs = []
        for inputs in engine_inputs:
            engine_outputs.append(self.engine(inputs))

        return engine_outputs

    def process_inputs(
        self,
        inputs: QuestionAnsweringInput,
    ) -> Tuple[List[numpy.ndarray], Dict[str, Any]]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            QuestionAnsweringInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine and
            dictionary of parsed features and original extracted example
        """
        squad_example = SquadExample(
            None, inputs.question, inputs.context, None, None, None
        )
        features = self._tokenize(squad_example)
        tokens = [f.__dict__ for f in features]

        engine_inputs_ = [self.tokens_to_engine_input(t) for t in tokens]
        # add batch dimension, assuming batch size 1
        engine_inputs = []
        for inps in engine_inputs_:
            engine_inputs.append([numpy.expand_dims(inp, axis=0) for inp in inps])

        return engine_inputs, dict(features=features, example=squad_example)

    def process_engine_outputs(
        self,
        engine_outputs: List[List[numpy.ndarray]],
        **kwargs,
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """

        features = kwargs["features"]
        example = kwargs["example"]

        # Loop through all features
        # Selects the feature with highest score
        score = -float("Inf")
        ans_start = None
        ans_end = None
        feature = None
        for feature_, outputs in zip(features, engine_outputs):
            ans_start_, ans_end_, score_ = self._process_single_feature(
                feature_, outputs
            )
            if score_ > score:
                score = score_
                ans_start = ans_start_
                ans_end = ans_end_
                feature = feature_

        # decode start, end idx into text
        if not self.tokenizer.is_fast:
            char_to_word = numpy.array(example.char_to_word_offset)
            return self.output_schema(
                score=score.item(),
                start=numpy.where(char_to_word == feature.token_to_orig_map[ans_start])[
                    0
                ][0].item(),
                end=numpy.where(char_to_word == feature.token_to_orig_map[ans_end])[0][
                    -1
                ].item(),
                answer=" ".join(
                    example.doc_tokens[
                        feature.token_to_orig_map[
                            ans_start
                        ] : feature.token_to_orig_map[ans_end]
                        + 1
                    ]
                ),
            )
        else:
            question_first = bool(self.tokenizer.padding_side == "right")

            # Sometimes the max probability token is in the middle of a word so:
            # we start by finding the right word containing the token with
            # `token_to_word` then we convert this word in a character span

            # If the start or end token point to the separator token
            # move the token by one
            def _token_to_char(token):
                w = feature.encoding.token_to_word(token)
                if w is None:
                    return None
                else:
                    return feature.encoding.word_to_chars(
                        w, sequence_index=1 if question_first else 0
                    )

            char_start = _token_to_char(ans_start)
            while char_start is None:
                ans_start += 1
                char_start = _token_to_char(ans_start)

            char_end = _token_to_char(ans_end)
            while char_end is None:
                ans_end += 1
                char_end = _token_to_char(ans_end)

            return self.output_schema(
                score=score.item(),
                start=char_start[0],
                end=char_end[1],
                answer=example.context_text[char_start[0] : char_end[1]],
            )

    def _process_single_feature(self, feature, engine_outputs):
        """
        :param feature: a SQuAD feature object
        :param engine_outputs: logits for start and end tokens
        :return: answer start token (int), answer end token (int), score (float)
        """
        start_vals, end_vals = engine_outputs[:2]

        # assuming batch size 0
        start = start_vals[0]
        end = end_vals[0]

        # Ensure padded tokens & question tokens cannot belong
        undesired_tokens = (
            numpy.abs(numpy.array(feature.p_mask) - 1) & feature.attention_mask
        )

        # Generate mask
        undesired_tokens_mask = undesired_tokens == 0.0

        # Make sure non-context indexes cannot contribute to the softmax
        start = numpy.where(undesired_tokens_mask, -10000.0, start)
        end = numpy.where(undesired_tokens_mask, -10000.0, end)

        # Normalize logits and spans to retrieve the answer
        start = numpy.exp(
            start - numpy.log(numpy.sum(numpy.exp(start), axis=-1, keepdims=True))
        )
        end = numpy.exp(
            end - numpy.log(numpy.sum(numpy.exp(end), axis=-1, keepdims=True))
        )

        # Mask CLS
        start[0] = 0.0
        end[0] = 0.0

        ans_start, ans_end, scores = self._decode(start, end)
        # assuming one stride, so grab first idx
        ans_start = ans_start[0]
        ans_end = ans_end[0]
        score = scores[0]

        return ans_start, ans_end, score

    def _tokenize(self, example: SquadExample):
        if not self.tokenizer.is_fast:
            features = squad_convert_examples_to_features(
                examples=[example],
                tokenizer=self.tokenizer,
                max_set_length=self.sequence_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_question_length,
                padding_strategy=PaddingStrategy.MAX_LENGTH.value,
                is_training=False,
                tqdm_enabled=False,
            )
        else:
            question_first = bool(self.tokenizer.padding_side == "right")
            encoded_inputs = self.tokenizer(
                text=example.question_text if question_first else example.context_text,
                text_pair=(
                    example.context_text if question_first else example.question_text
                ),
                padding=PaddingStrategy.MAX_LENGTH.value,
                truncation="only_second" if question_first else "only_first",
                max_length=self.sequence_length,
                stride=self.doc_stride,
                return_tensors="np",
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )

            # p_mask: mask with 1 for token than cannot be in the answer
            # We put 0 on the tokens from the context and 1 everywhere else
            p_mask = numpy.asarray(
                [
                    [
                        tok != 1 if question_first else 0
                        for tok in encoded_inputs.sequence_ids(0)
                    ]
                ]
            )

            # keep the cls_token unmasked
            if self.tokenizer.cls_token_id is not None:
                cls_index = numpy.nonzero(
                    encoded_inputs["input_ids"][0] == self.tokenizer.cls_token_id
                )
                p_mask[cls_index] = 0

            features = []
            for i in range(len(encoded_inputs["input_ids"])):
                features.append(
                    SquadFeatures(
                        input_ids=encoded_inputs["input_ids"][i],
                        attention_mask=encoded_inputs["attention_mask"][i],
                        token_type_ids=encoded_inputs["token_type_ids"][i],
                        p_mask=p_mask[0].tolist(),
                        encoding=encoded_inputs[i],
                        # the following values are unused for fast tokenizers
                        cls_index=None,
                        token_to_orig_map={},
                        example_index=0,
                        unique_id=0,
                        paragraph_len=0,
                        token_is_max_context=0,
                        tokens=[],
                        start_position=0,
                        end_position=0,
                        is_impossible=False,
                        qas_id=None,
                    )
                )

        return features

    def _decode(self, start: numpy.ndarray, end: numpy.ndarray) -> Tuple:
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = numpy.matmul(numpy.expand_dims(start, -1), numpy.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = numpy.tril(numpy.triu(outer), self.max_answer_length - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        # only returning best result, use argsort for topk support
        idx_sort = [numpy.argmax(scores_flat)]

        start, end = numpy.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end]
