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

import json
from typing import Dict, List, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse.v2.operators import Operator


class ImageClassificationOutput(BaseModel):
    """
    Output model for image classification
    """

    labels: List[Union[int, str, List[int], List[str]]] = Field(
        description="List of labels, one for each prediction"
    )
    scores: List[Union[float, List[float]]] = Field(
        description="List of scores, one for each prediction"
    )


__all__ = ["ImageClassificationPostProcess"]


class ImageClassificationPostProcess(Operator):
    """
    Image Classification post-processing Operator. This Operator is responsible for
    processing outputs from the engine and returning the classification results to
    the user, using the ImageClassifcationOutput structure.
    """

    input_schema = None
    output_schema = ImageClassificationOutput

    def __init__(
        self, top_k: int = 1, class_names: Union[None, str, Dict[str, str]] = None
    ):
        self.top_k = top_k
        if isinstance(class_names, str) and class_names.endswith(".json"):
            self._class_names = json.load(open(class_names))
        elif isinstance(class_names, dict):
            self._class_names = class_names
        else:
            self._class_names = None

    def run(self, inp: "EngineOperatorOutputs", **kwargs) -> Dict:  # noqa: F821
        labels, scores = [], []
        inp = inp.engine_outputs
        for prediction_batch in inp[0]:
            label = (-prediction_batch).argsort()[: self.top_k]
            score = prediction_batch[label]
            labels.append(label)
            scores.append(score.tolist())

        if self._class_names is not None:
            labels = numpy.vectorize(self._class_names.__getitem__)(labels)
            labels = labels.tolist()

        if isinstance(labels[0], numpy.ndarray):
            labels = [label.tolist() for label in labels]

        if len(labels) == 1:
            labels = labels[0]
            scores = scores[0]

        return {"scores": scores, "labels": labels}
