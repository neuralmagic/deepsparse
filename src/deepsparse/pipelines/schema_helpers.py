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
from typing import Generator


__all__ = [
    "Splittable",
    "Joinable",
]


class Splittable(ABC):
    """
    A contract that ensures implementing subclass objects(representing a batch
    size of b) can be split into a smaller List of objects each representing a input
    of batch size 1
    """

    @abstractmethod
    def split(
        self,
        *args,
        **kwargs,
    ) -> Generator["BaseModel", None, None]:  # noqa: F821
        """
        Utility abstract method that subclasses must implement, the goal of
        this function is to split a Schema object with a batch size b, into a
        generator of b smaller Schema objects with batch size 1, the returned
        object can be iterated on.

        :return: A Generator of smaller objects each representing an input of
            batch-size 1
        """
        raise NotImplementedError


class Joinable(ABC):
    """
    A contract that ensures multiple objects of the implementing subclass can be
    combined into one object representing a bigger batch size
    """

    @staticmethod
    @abstractmethod
    def join(self, *args, **kwargs) -> "BaseModel":  # noqa: F821
        """
        Utility abstract method that subclasses must implement, the goal of
        this function is to take in an Iterable of subclass objects and combine
        them into one object representing a bigger batch size

        :return: A JoinableSchema object that represents a bigger batch
        """
        raise NotImplementedError
