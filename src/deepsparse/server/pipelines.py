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

"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from deepsparse.server.config import ServeModelConfig, ServerConfig
from deepsparse.tasks import SupportedTasks


__all__ = ["PipelineDefinition", "load_pipelines_definitions"]


class PipelineDefinition(BaseModel):
    pipeline: Any = Field(description=(""))
    request_model: Any = Field(description=(""))
    response_model: Any = Field(description=(""))
    kwargs: Dict[str, Any] = Field(description=(""))
    config: ServeModelConfig = Field(description=(""))


def load_pipelines_definitions(config: ServerConfig) -> List[PipelineDefinition]:
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
