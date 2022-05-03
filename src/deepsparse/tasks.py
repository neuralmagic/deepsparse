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
Classes and implementations for supported tasks in the DeepSparse pipeline and system
"""

from collections import namedtuple
from typing import List


__all__ = ["SupportedTasks", "AliasedTask"]


class AliasedTask:
    """
    A task that can have multiple aliases to match to.
    For example, question_answering which can alias to qa as well

    :param name: the name of the task such as question_answering or text_classification
    :param aliases: the aliases the task can go by in addition to the name such as
        qa, glue, sentiment_analysis, etc
    """

    def __init__(self, name: str, aliases: List[str]):
        self._name = name
        self._aliases = aliases

    @property
    def name(self) -> str:
        """
        :return: the name of the task such as question_answering
        """
        return self._name

    @property
    def aliases(self) -> List[str]:
        """
        :return: the aliases the task can go by such as qa, glue, sentiment_analysis
        """
        return self._aliases

    def matches(self, task: str) -> bool:
        """
        :param task: the name of the task to check whether the given instance matches.
            Checks the current name as well as any aliases.
            Everything is compared at lower case and "-" are replaced with "_".
        :return: True if task does match the current instance, False otherwise
        """
        task = task.lower().replace("-", "_")

        return task == self.name or task in self.aliases


class SupportedTasks:
    """
    The supported tasks in the DeepSparse pipeline and system
    """

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
        """
        :param task: the name of the task to check whether it is an nlp task
            such as question_answering
        :return: True if it is an nlp task, False otherwise
        """
        return (
            cls.nlp.question_answering.matches(task)
            or cls.nlp.text_classification.matches(task)
            or cls.nlp.token_classification.matches(task)
        )
