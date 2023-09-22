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

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


__all__ = ["PipelineInputType", "PipelineBenchmarkConfig"]


class PipelineInputType:
    DUMMY: str = "dummy"
    REAL: str = "real"


class PipelineBenchmarkConfig(BaseModel):
    data_type: str = Field(
        default=PipelineInputType.DUMMY,
        description=(
            "Type of data source, dummy to generate data or real to load from file."
        ),
    )

    gen_sequence_length: Optional[int] = Field(
        default=512,
        description=(
            "Number of characters to generate for pipelines that take text input."
        ),
    )

    input_image_shape: Optional[List[int]] = Field(
        default=[224, 224, 3],
        description=(
            "Image size for pipelines that take image input, 3-dim with channel as the "
            "last dimmension"
        ),
    )

    data_folder: Optional[str] = Field(
        default=None,
        description=(
            "Path to local folder of input data containing text or image files"
        ),
    )

    recursive_search: bool = Field(
        default=False,
        description=("whether to recursively search through data_folder for files"),
    )

    max_string_length: int = Field(
        default=-1,
        description=(
            "Maximum characters to read from each text file, -1 for no maximum"
        ),
    )

    question_file: Optional[str] = Field(
        default=None, description=("Path to text file to read question from")
    )

    context_file: Optional[str] = Field(
        default=None, description=("Path to text file to read question context from")
    )

    pipeline_kwargs: Dict = Field(
        default={}, description=("Additional arguments passed to pipeline creation")
    )

    input_schema_kwargs: Dict = Field(
        default={},
        description=("Additional arguments passed to input schema creations "),
    )
