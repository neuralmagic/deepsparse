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

import logging
from typing import Dict, Optional, Tuple, Union

from deepsparse.v2.image_classification.postprocess_operator import (
    ImageClassificationPostProcess,
)
from deepsparse.v2.image_classification.preprocess_operator import (
    ImageClassificationPreProcess,
)
from deepsparse.v2.operators.engine_operator import EngineOperator
from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.routers.router import LinearRouter
from deepsparse.v2.schedulers.scheduler import OperatorScheduler


_LOGGER = logging.getLogger(__name__)

__all__ = ["ImageClassificationPipeline"]


class ImageClassificationPipeline(Pipeline):
    def __init__(
        self,
        engine_kwargs: Dict,
        class_names: Union[None, str, Dict[str, str]] = None,
        image_size: Optional[Tuple[int]] = None,
        top_k: int = 1,
    ):
        if not engine_kwargs.get("model_path"):
            raise ValueError("engine_kwargs must include model_path path")

        engine = EngineOperator(**engine_kwargs)
        preproces = ImageClassificationPreProcess(
            model_path=engine.model_path, image_size=image_size
        )
        postprocess = ImageClassificationPostProcess(
            top_k=top_k, class_names=class_names
        )

        ops = [preproces, engine, postprocess]
        router = LinearRouter(end_route=len(ops))
        scheduler = [OperatorScheduler()]
        super().__init__(ops=ops, router=router, schedulers=scheduler)
