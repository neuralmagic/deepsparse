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

from typing import Any, Iterable, List, TextIO, Union

import numpy


try:
    from PIL import Image

    pil_import_error = None
except Exception as import_error:
    Image, pil_import_error = None, import_error

from pydantic import BaseModel, Field


__all__ = [
    "ComputerVisionSchema",
]


class ComputerVisionSchema(BaseModel):
    """
    A base ComputerVisionSchema to accept images, it is recommended to inherit
    ComputerVisionSchema for all Computer Vision Based tasks, this Schema provides a
    `from_files` factory method, and also specifies Field types for images
    """

    images: Union[str, List[str], List[Any], Any] = Field(
        description="List of Images to process"
    )  # List[Any] to accept List[numpy.ndarray], Any to accept numpy.ndarray

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_files(
        cls,
        files: Iterable[TextIO],
        *args,
        from_server: bool = False,
        **kwargs,
    ) -> BaseModel:
        """
        :param files: Iterable of file pointers to create ImageClassificationInput from
        :return: ImageClassificationInput constructed from files
        """
        if pil_import_error is not None:
            raise ImportError(
                "PIL is a requirement for Computer Vision pipeline schemas,"
                f" but was not found. Error:\n{pil_import_error}, "
                "try `pip install Pillow`"
            )
        images = [numpy.asarray(Image.open(file)) for file in files]
        return cls(*args, images=images, **kwargs)
