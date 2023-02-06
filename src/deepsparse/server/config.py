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

from pydantic import BaseModel, Field, validator

from deepsparse import DEEPSPARSE_ENGINE, PipelineConfig
from deepsparse.loggers.config import (
    MetricFunctionConfig,
    SystemLoggingConfig,
    SystemLoggingGroup,
)
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


class ServerSystemLoggingConfig(SystemLoggingConfig):
    """
    Extends the `SystemLoggingConfig` schema to include system group metrics
    that pertain to the Server.
    """

    request_details: SystemLoggingGroup = Field(
        default=SystemLoggingGroup(enable=False),
        description="The configuration group for the request_details system "
        "logging group. For details refer to the DeepSparse server "
        "system logging documentation. By default this group is disabled.",
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

    data_logging: Optional[Dict[str, List[MetricFunctionConfig]]] = Field(
        default=None,
        description="Specifies the rules for the data logging. "
        "It relates a key (name of the logging target) "
        "to a list of metric functions that are to be applied"
        "to this target prior to logging.",
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

        engine_type = kwargs.pop("engine_type", DEEPSPARSE_ENGINE)

        return PipelineConfig(
            task=self.task,
            model_path=self.model,
            engine_type=engine_type,
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

    loggers: Dict[str, Optional[Dict[str, Any]]] = Field(
        default={},
        description=(
            "Optional dictionary of logger integration names to initialization kwargs."
            "Set to {} for no loggers. Default is {}."
        ),
    )

    system_logging: ServerSystemLoggingConfig = Field(
        default_factory=ServerSystemLoggingConfig,
        description="A model that holds the system logging configuration. "
        "If not specified explicitly in the yaml config, the "
        "default SystemLoggingConfig model is used.",
    )

    @validator("endpoints")
    def assert_unique_endpoint_names(
        cls, endpoints: List[EndpointConfig]
    ) -> List[EndpointConfig]:
        name_list = []
        for endpoint in endpoints:
            name = endpoint.name
            if name is None:
                continue
            if name in name_list:
                raise ValueError(
                    "Endpoint names must be unique if specified. "
                    "Found a duplicated endpoint name: {}".format(name)
                )
            name_list.append(name)
        return endpoints

    @validator("endpoints")
    def set_unique_endpoint_names(
        cls, endpoints: List[EndpointConfig]
    ) -> List[EndpointConfig]:
        """
        Assert that the endpoints in ServerConfig have unique names.
        If endpoint does not have a `name` specified, the endpoint is
        named `{task_name}-{idx}`.

        :param endpoints: configuration of server's endpoints
        :return: configuration of server's endpoints
        """
        counter_task_name_used = {endpoint.task: 0 for endpoint in endpoints}
        # make sure that the endpoints in ServerConfig have unique names.
        for endpoint_config in endpoints:
            if endpoint_config.name is None:
                task_name = endpoint_config.task
                idx = counter_task_name_used[task_name]
                counter_task_name_used[task_name] += 1
                endpoint_config.name = f"{endpoint_config.task}-{idx}"
        return endpoints


def endpoint_diff(
    old_cfg: ServerConfig, new_cfg: ServerConfig
) -> Tuple[List[EndpointConfig], List[EndpointConfig]]:
    """
    - Added endpoint: the endpoint's route is **not** present in `old_cfg`,
        and present in `new_cfg`.
    - Removed endpoint: the endpoint's route is present in `old_cfg`, and
        **not** present in `new_cfg`.
    - Modified endpoint: Any field of the endpoint changed. In this case
        the endpoint will be present in both returned lists (it is both
        added and removed).
    :return: Tuple of (added endpoints, removed endpoints).
    """
    routes_in_old = {
        endpoint.route: endpoint
        for endpoint in old_cfg.endpoints
        if endpoint.route is not None
    }
    routes_in_new = {
        endpoint.route: endpoint
        for endpoint in new_cfg.endpoints
        if endpoint.route is not None
    }

    added_routes = set(routes_in_new) - set(routes_in_old)
    removed_routes = set(routes_in_old) - set(routes_in_new)

    # for any routes that are in both, check if the config object is different.
    # if so, then we do modification by adding the route to both remove & add
    for route in set(routes_in_new) & set(routes_in_old):
        if routes_in_old[route] != routes_in_new[route]:
            removed_routes.add(route)
            added_routes.add(route)

    added_endpoints = [
        endpoint for endpoint in new_cfg.endpoints if endpoint.route in added_routes
    ]
    removed_endpoints = [
        endpoint for endpoint in old_cfg.endpoints if endpoint.route in removed_routes
    ]
    return added_endpoints, removed_endpoints


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
