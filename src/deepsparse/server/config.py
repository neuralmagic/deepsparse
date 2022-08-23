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


from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field


__all__ = [
    "ServerConfig",
    "EndpointConfig",
    "SequenceLengthsConfig",
    "ImageSizesConfig",
]


class SequenceLengthsConfig(BaseModel):
    sequence_lengths: List[int] = Field(
        description="The sequence lengths the model should accept"
    )


class ImageSizesConfig(BaseModel):
    image_sizes: List[Tuple[int, int]] = Field(
        description="The list of image sizes the model should accept"
    )


class EndpointConfig(BaseModel):
    name: str = Field(
        description="Name of the model used for logging & metric purposes."
    )

    endpoint: str = Field(
        description="The path to use for this endpoint. E.g. '/predict'."
    )

    task: str = Field(description="Task this endpoint performers")

    model: str = Field(description="Location of the underlying model to use.")

    batch_size: int = Field(description="The batch size to compile the model for.")

    accept_multiples_of_batch_size: bool = Field(
        description="""
Whether to accept any request with a batch size that is a multiple of `batch_size`.

E.g. if batch_size is 1 and this field is True,
then the model can accept any batch size.
    """,
    )

    bucketing: Optional[Union[ImageSizesConfig, SequenceLengthsConfig]] = Field(
        default=None,
        description="What input shapes this model can accept. Must specify at least 1",
    )


class ServerConfig(BaseModel):
    num_cores: int = Field(
        description="The number of cores available for model execution. "
        "Defaults to all available cores.",
    )

    num_concurrent_batches: int = Field(
        description="""
The number of times to partition the available cores. Each batch will have access to
`num_cores / num_concurrent_batches` cores.
""",
    )

    endpoints: List[EndpointConfig] = Field(description="The models to serve.")
