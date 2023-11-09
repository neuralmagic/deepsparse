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

from typing import Union

from transformers import AutoModelForCausalLM

from deepsparse import DEEPSPARSE_ENGINE, ORT_ENGINE, Pipeline


__all__ = ["initialize_model_from_target"]


def initialize_model_from_target(
    target: str, engine_type: str, **kwargs
) -> Union[Pipeline, AutoModelForCausalLM]:
    """
    :param target: The target path to initialize the model from. This can be
        a local or remote path to the model or a sparsezoo stub
    :param engine_type: The engine type to initialize the model with.
    :param kwargs: Additional kwargs to pass to the model initialization
    :return: The initialized model
    """
    if engine_type in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        return Pipeline.create(
            task="text-generation", model_path=target, engine_type=engine_type, **kwargs
        )
    elif engine_type == "torch":
        return AutoModelForCausalLM.from_pretrained(target, **kwargs)
    else:
        raise NotImplementedError("Unsupported engine type")
