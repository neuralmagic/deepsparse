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


from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from deepsparse import DEEPSPARSE_ENGINE, PipelineConfig
from deepsparse.tasks import SupportedTasks


__all__ = [
    "ServerConfig",
    "EndpointConfig",
    "SequenceLengthsConfig",
    "ImageSizesConfig",
]

# these are stored as global variables instead of enum because in order
# to save/load enums using yaml, you have to enable arbitrary code
# execution.
INTEGRATION_LOCAL = "local"
INTEGRATION_SAGEMAKER = "sagemaker"
INTEGRATIONS = [INTEGRATION_LOCAL, INTEGRATION_SAGEMAKER]


class SequenceLengthsConfig(BaseModel):
    sequence_lengths: List[int] = Field(
        description="The sequence lengths the model should accept"
    )


class ImageSizesConfig(BaseModel):
    image_sizes: List[Tuple[int, int]] = Field(
        description="The list of image sizes the model should accept"
    )


class EndpointConfig(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the model used for logging & metric purposes. "
            "If not specified 'endpoint-<index>' will be used."
        ),
    )

    route: Optional[str] = Field(
        default=None,
        description="Optional url to use for this endpoint. E.g. '/predict'. "
        "If there are multiple endpoints, all routes must be specified. "
        "If there is a single endpoint, '/predict' is default if not specified.",
    )

    task: str = Field(description="Task this endpoint performs")

    model: str = Field(description="Location of the underlying model to use.")

    batch_size: int = Field(
        default=1, description="The batch size to compile the model for."
    )

    bucketing: Optional[Union[ImageSizesConfig, SequenceLengthsConfig]] = Field(
        default=None,
        description=(
            "What input shapes this model can accept."
            "Example for multiple sequence lengths in yaml: "
            "```yaml\n"
            "bucketing:\n"
            "  sequence_lengths: [16, 32, 64]\n"
            "```\n"
        ),
    )

    kwargs: Dict[str, Any] = Field(
        default={}, description="Additional arguments to pass to the Pipeline"
    )

    def to_pipeline_config(self) -> PipelineConfig:
        input_shapes, kwargs = _unpack_bucketing(self.task, self.bucketing)

        kwargs.update(self.kwargs)

        return PipelineConfig(
            task=self.task,
            model_path=self.model,
            engine_type=DEEPSPARSE_ENGINE,
            batch_size=self.batch_size,
            num_cores=None,  # this will be set from Context
            alias=self.name,
            input_shapes=input_shapes,
            kwargs=kwargs,
        )


class ServerConfig(BaseModel):
    num_cores: Optional[int] = Field(
        description="The number of cores available for model execution. "
        "Defaults to all available cores.",
        default=None,
    )

    num_workers: Optional[int] = Field(
        description="The number of workers to split the available cores between. "
        "Defaults to half of the num_cores set",
        default=None,
    )

    integration: str = Field(
        default=INTEGRATION_LOCAL,
        description="The kind of integration to use. local|sagemaker",
    )

    engine_thread_pinning: str = Field(
        default="core",
        description=(
            "Enable binding threads to cores ('core' the default), "
            "threads to cores on sockets ('numa'), or disable ('none')"
        ),
    )

    pytorch_num_threads: Optional[int] = Field(
        default=1,
        description=(
            "Configures number of threads that pytorch is allowed to use during"
            "pre and post-processing. Useful to reduce resource contention. "
            "Set to `None` to place no restrictions on pytorch."
        ),
    )

    endpoints: List[EndpointConfig] = Field(description="The models to serve.")


def _unpack_bucketing(
    task: str, bucketing: Optional[Union[SequenceLengthsConfig, ImageSizesConfig]]
) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """
    :return: (input_shapes, kwargs) which are passed to PipelineConfig
    """
    if bucketing is None:
        return None, {}

    if isinstance(bucketing, SequenceLengthsConfig):
        if not SupportedTasks.is_nlp(task):
            raise ValueError(f"SequenceLengthConfig specified for non-nlp task {task}")

        return _unpack_nlp_bucketing(bucketing)
    elif isinstance(bucketing, ImageSizesConfig):
        if not SupportedTasks.is_cv(task):
            raise ValueError(
                f"ImageSizeConfig specified for non computer vision task {task}"
            )

        return _unpack_cv_bucketing(bucketing)
    else:
        raise ValueError(f"Unknown bucket config {bucketing}")


def _unpack_nlp_bucketing(cfg: SequenceLengthsConfig):
    if len(cfg.sequence_lengths) == 0:
        raise ValueError("Must specify at least one sequence length under bucketing")

    if len(cfg.sequence_lengths) == 1:
        input_shapes = None
        kwargs = {"sequence_length": cfg.sequence_lengths[0]}
    else:
        input_shapes = None
        kwargs = {"sequence_length": cfg.sequence_lengths}

    return input_shapes, kwargs


def _unpack_cv_bucketing(cfg: ImageSizesConfig):
    if len(cfg.image_sizes) == 0:
        raise ValueError("Must specify at least one image size under bucketing")

    if len(cfg.image_sizes) == 1:
        # NOTE: convert from List[Tuple[int, int]] to List[List[int]]
        input_shapes = [list(cfg.image_sizes[0])]
        kwargs = {}
    else:
        raise NotImplementedError(
            "Multiple image size buckets is currently unsupported"
        )

    return input_shapes, kwargs
