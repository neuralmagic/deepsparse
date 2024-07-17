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

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from deepsparse.operators.engine_operator import DEEPSPARSE_ENGINE


class PipelineConfig(BaseModel):
    """
    Configuration for creating a Pipeline object

    Can be used to create a Pipeline from a config object or file with
    Pipeline.from_config(), or used as a building block for other configs
    such as for deepsparse.server
    """

    task: str = Field(
        description="name of task to create a pipeline for",
    )
    model_path: str = Field(
        default=None,
        description="path on local system or SparseZoo stub to load the model from",
    )
    engine_type: Optional[str] = Field(
        default=DEEPSPARSE_ENGINE,
        description=(
            "inference engine to use. Currently supported values include "
            "'deepsparse' and 'onnxruntime'. Default is 'deepsparse'"
        ),
    )
    batch_size: Optional[int] = Field(
        default=1,
        description=("static batch size to use for inference. Default is 1"),
    )
    num_cores: Optional[int] = Field(
        default=None,
        description=(
            "number of CPU cores to allocate for inference engine. None"
            "specifies all available cores. Default is None"
        ),
    )
    scheduler: Optional[str] = Field(
        default="async",
        description=(
            "(deepsparse only) kind of scheduler to execute with. Defaults to async"
        ),
    )
    input_shapes: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "list of shapes to set ONNX the inputs to. Pass None to use model as-is. "
            "Default is None"
        ),
    )
    alias: Optional[str] = Field(
        default=None,
        description=(
            "optional name to give this pipeline instance, useful when inferencing "
            "with multiple models. Default is None"
        ),
    )
    middlewares: Optional[List[str]] = Field(
        default=None,
        description="Middlewares to use",
    )
    kwargs: Optional[Dict[str, Any]] = Field(
        default={},
        description=(
            "Additional arguments for inference with the model that will be passed "
            "into the pipeline as kwargs"
        ),
    )

    # override name spaces due to model_ warnings in pydantic 2.X
    model_config = ConfigDict(protected_namespaces=())
