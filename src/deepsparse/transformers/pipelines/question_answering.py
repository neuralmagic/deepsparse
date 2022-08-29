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

import collections
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy
from pydantic import BaseModel, Field
from transformers.data import SquadExample

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "QuestionAnsweringInput",
    "QuestionAnsweringOutput",
    "QuestionAnsweringPipeline",
]

_LOGGER = logging.getLogger(__name__)


class QuestionAnsweringInput(BaseModel):
    """
    Schema for inputs to question_answering pipelines
    """

    question: str = Field(description="String question to be answered")
    context: str = Field(description="String representing context for answer")
    id: str = Field(description="Sample identifier", default=None)


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
    :param doc_stride: if the context is too long to fit with the question for the
        model, it will be split in several chunks with some overlap. This argument
        controls the size of that overlap. Default is 128
    :param max_question_length: maximum length of the question after tokenization.
        It will be truncated if needed. Default is 64
    :param max_answer_length: maximum length of answer after decoding. Default is 15
    :param n_best_size: number of n-best predictions to generate when looking for
        an answer. Default is 20
    :param pad_to_max_length: whether to pad all samples to max sequence length.
        If False, will pad the samples dynamically when batching to the maximum length
        in the batch
    :param version_2_with_negative: if true, some examples do not have an answer
    :param output_dir: output folder to save predictions, used for debugging
    :param num_spans: if the context is too long to fit with the question for the
        model, it will be split in several chunks. This argument controls the maximum
        number of spans to feed into the model.
    """

    def __init__(
        self,
        *,
        doc_stride: int = 128,
        max_question_length: int = 64,
        max_answer_length: int = 15,
        n_best_size: int = 20,
        pad_to_max_length: bool = True,
        version_2_with_negative: bool = False,
        output_dir: str = None,
        num_spans: Optional[int] = None,
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
        self._n_best_size = n_best_size
        self._pad_to_max_length = pad_to_max_length
        self._version_2_with_negative = version_2_with_negative
        self._output_dir = output_dir
        self._num_spans = num_spans

        super().__init__(**kwargs)

    @property
    def doc_stride(self) -> int:
        """
        :return: if the context is too long to fit with the question for the
            model, it will be split in several chunks with some overlap. This argument
            controls the size of that overlap.
        """
        return self._doc_stride

    @property
    def max_answer_length(self) -> int:
        """
        :return: maximum length of answer after decoding
        """
        return self._max_answer_length

    @property
    def n_best_size(self) -> int:
        """
        :return: The total number of n-best predictions to generate when looking
            for an answer
        """
        return self._n_best_size

    @property
    def pad_to_max_length(self) -> int:
        """
        :return: whether to pad all samples to max_sequence_length
        """
        return self._pad_to_max_length

    @property
    def max_question_length(self) -> int:
        """
        :return: maximum length of the question after tokenization.
            It will be truncated if needed
        """
        return self._max_question_length

    @property
    def version_2_with_negative(self) -> bool:
        """
        :return: Whether or not the underlying dataset contains examples with
            no answers
        """
        return self._version_2_with_negative

    @property
    def output_dir(self) -> str:
        """
        :return: path to output folder for predictions
        """
        return self._output_dir

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
            inputs.id, inputs.question, inputs.context, None, None, None
        )
        tokenized_example = self._tokenize(squad_example)

        span_engine_inputs = []
        span_extra_info = []
        num_spans = len(tokenized_example["input_ids"])
        for span in range(num_spans):
            span_input = [
                numpy.array(tokenized_example[key][span])
                for key in self.onnx_input_names
            ]
            span_engine_inputs.append(span_input)

            span_extra_info.append(
                {
                    key: numpy.array(tokenized_example[key][span])
                    for key in tokenized_example.keys()
                    if key not in self.onnx_input_names
                }
            )

        # add batch dimension, assuming batch size 1
        engine_inputs = list(map(numpy.stack, zip(*span_engine_inputs)))

        return engine_inputs, dict(
            span_extra_info=span_extra_info, example=squad_example
        )

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
        span_extra_info = kwargs["span_extra_info"]
        example = kwargs["example"]
        num_spans = len(span_extra_info)

        if len(engine_outputs) != 2:
            raise ValueError(
                "`engine_outputs` should be a list with two elements "
                "[start_logits, end_logits]."
            )

        all_start_logits, all_end_logits = engine_outputs

        if (
            all_start_logits.shape[0] != num_spans
            or all_end_logits.shape[0] != num_spans
        ):
            raise ValueError(
                "Engine outputs expected for {num_spans} span(s), "
                "but found for {len(engine_outputs)}"
            )

        if self.version_2_with_negative:
            scores_diff_json = collections.OrderedDict()
            null_score_diff_threshold = 0.0

        min_null_prediction = None
        prelim_predictions = []
        for span_idx in range(num_spans):
            start_logits = all_start_logits[span_idx]
            end_logits = all_end_logits[span_idx]

            # This is what will allow us to map some the positions in our logits to
            # span of texts in the original context.
            offset_mapping = span_extra_info[span_idx]["offset_mapping"]

            # Optional `token_is_max_context`, if provided we will remove answers
            # that do not have the maximum context available in the current feature.
            token_is_max_context = span_extra_info[span_idx].get(
                "token_is_max_context", None
            )

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `self.n_best_size` greater start
            # and end logits.
            start_indexes = numpy.argsort(start_logits)[
                -1 : -self.n_best_size - 1 : -1
            ].tolist()
            end_indexes = numpy.argsort(end_logits)[
                -1 : -self.n_best_size - 1 : -1
            ].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices
                    # are out of bounds or correspond to part of the input_ids that
                    # are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is
                    # either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > self.max_answer_length
                    ):
                        continue
                    # Don't consider answer that don't have the maximum context
                    # available (if such information is provided).
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if self.version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `self.n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[: self.n_best_size]

        # Add back the minimum null prediction if it was removed because of its
        # low score.
        if self.version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        best_start, best_end = predictions[0]["offsets"]
        best_score = predictions[0]["score"]

        # Use the offsets to gather the answer text in the original context.
        context = example.context_text
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we
        # create a fake prediction to avoid failure
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # Compute the softmax of all scores (we do it with numpy to stay independent
        # from torch/tf in this file, using the LogSumExp trick)
        scores = numpy.array([pred.pop("score") for pred in predictions])
        exp_scores = numpy.exp(scores - numpy.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not self.version_2_with_negative:
            all_predictions[example.qas_id] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example.qas_id] = float(
                score_diff
            )  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting numpy.float back to float.
        all_nbest_json[example.qas_id] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (numpy.float16, numpy.float32, numpy.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

        if self.output_dir is not None:
            scores_diff_json = (
                None if not self.version_2_with_negative else scores_diff_json
            )
            self._save_predictions(all_predictions, all_nbest_json, scores_diff_json)

        return self.output_schema(
            score=best_score,
            start=best_start,
            end=best_end,
            answer=example.context_text[best_start:best_end],
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
            " ".join((input_schema.context, input_schema.question)),
            add_special_tokens=True,
            return_tensors="np",
            padding=False,
            truncation=False,
        )
        input_seq_len = max(map(len, tokens["input_ids"]))
        return TransformersPipeline.select_bucket_by_seq_len(input_seq_len, pipelines)

    def _tokenize(self, example: SquadExample, *args):
        # The logic here closely matches the tokenization step performed
        # on evaluation dataset in the SparseML question answering training script
        if not self.tokenizer.is_fast:
            raise ValueError(
                "This example script only works for models that have a fast tokenizer."
            )
        else:
            pad_on_right = self.tokenizer.padding_side == "right"
            tokenized_example = self.tokenizer(
                text=example.question_text if pad_on_right else example.context_text,
                text_pair=(
                    example.context_text if pad_on_right else example.question_text
                ),
                truncation="only_second" if pad_on_right else "only_first",
                max_length=self.sequence_length,
                stride=self.doc_stride,
                return_token_type_ids=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                padding="max_length" if self.pad_to_max_length else False,
            )

            # For evaluation, we will need to convert our predictions to substrings of
            # the context, so we keep the corresponding example_id and we will store
            # the offset mappings.
            tokenized_example["example_id"] = []

            n_spans = len(tokenized_example["input_ids"])
            for span in range(n_spans):
                # Grab the sequence corresponding to that example
                # (to know what is the context and what is the question).
                sequence_ids = tokenized_example.sequence_ids(span)
                context_index = 1 if pad_on_right else 0

                # Set to None the offset_mapping that are not part of the context so
                # it's easy to determine if a token position is part of the
                # context or not
                tokenized_example["offset_mapping"][span] = [
                    (ofmap if sequence_ids[key] == context_index else None)
                    for key, ofmap in enumerate(
                        tokenized_example["offset_mapping"][span]
                    )
                ]

                tokenized_example["example_id"].append(example.qas_id)

            if self._num_spans is not None:
                tokenized_example = {
                    k: tokenized_example[k][: self._num_spans]
                    for k in tokenized_example.keys()
                }

            return tokenized_example

    def _save_predictions(self, all_predictions, all_nbest_json, scores_diff_json):
        if not os.path.exists(self.output_dir):
            raise ValueError(f"Output folder {self.output_dir} not found.")

        if not os.path.isdir(self.output_dir):
            raise EnvironmentError(f"{self.output_dir} is not a directory.")

        prediction_file = os.path.join(self.output_dir, "predictions.json")
        nbest_file = os.path.join(self.output_dir, "nbest_predictions.json")
        if self.version_2_with_negative:
            null_odds_file = os.path.join(self.output_dir, "null_odds.json")

        mode = "a" if os.path.exists(prediction_file) else "w"
        with open(prediction_file, mode) as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        mode = "a" if os.path.exists(nbest_file) else "w"
        with open(nbest_file, mode) as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        if self.version_2_with_negative:
            mode = "a" if os.path.exists(null_odds_file) else "w"
            with open(null_odds_file, mode) as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
