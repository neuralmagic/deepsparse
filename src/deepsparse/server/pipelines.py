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
Pipelines that run preprocessing, postprocessing, and model inference
within the DeepSparse model server.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from deepsparse.server.config import ServeModelConfig, ServerConfig
from deepsparse.tasks import SupportedTasks


__all__ = ["PipelineDefinition", "load_pipelines_definitions"]


class PipelineDefinition(BaseModel):
    """
    A definition of a pipeline to be served by the model server.
    Used to create a prediction route on construction of the server app.
    """

    pipeline: Any = Field(description="the callable pipeline to invoke on each request")
    request_model: Any = Field(
        description="the pydantic model to validate the request body with"
    )
    response_model: Any = Field(
        description="the pydantic model to validate the response payload with"
    )
    kwargs: Dict[str, Any] = Field(
        description="any additional kwargs that should be passed into the pipeline"
    )
    config: ServeModelConfig = Field(
        description="the config for the model the pipeline is serving"
    )


def load_pipelines_definitions(config: ServerConfig) -> List[PipelineDefinition]:
    """
    Load the pipeline definitions to use for creating prediction routes from
    the given server configuration.

    :param config: the configuration to load pipeline definitions for
    :return: the loaded pipeline definitions to use for serving inference requests
    """
    defs = []

    for model_config in config.models:
        if SupportedTasks.is_nlp(model_config.task):
            # dynamically import so we don't install dependencies when unneeded
            from deepsparse.transformers.server import create_pipeline_definitions

            (
                pipeline,
                request_model,
                response_model,
                kwargs,
            ) = create_pipeline_definitions(model_config)
        else:
            raise ValueError(
                f"unsupported task given of {model_config.task} "
                f"for serve model config {model_config}"
            )

        defs.append(
            PipelineDefinition(
                pipeline=pipeline,
                request_model=request_model,
                response_model=response_model,
                kwargs=kwargs,
                config=model_config,
            )
        )

    return defs
