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

from collections import namedtuple
from enum import Enum
from typing import List


__all__ = ["SupportedTasks", "AliasedTask"]


class AliasedTask:
    def __init__(self, name: str, aliases: List[str]):
        self._name = name
        self._aliases = aliases

    @property
    def name(self) -> str:
        return self._name

    @property
    def aliases(self) -> List[str]:
        return self._aliases

    def matches(self, task: str) -> bool:
        return task == self.name or task in self.aliases


class SupportedTasks(Enum):
    nlp = namedtuple(
        "nlp", ["question_answering", "text_classification", "token_classification"]
    )(
        question_answering=AliasedTask("question_answering", ["qa"]),
        text_classification=AliasedTask(
            "text_classification", ["glue", "sentiment_analysis"]
        ),
        token_classification=AliasedTask("token_classification", ["ner"]),
    )

    @classmethod
    def is_nlp(cls, task: str) -> bool:
        return (
            cls.nlp.question_answering.matches(task)
            or cls.nlp.text_classification.matches(task)
            or cls.nlp.token_classification.matches(task)
        )
