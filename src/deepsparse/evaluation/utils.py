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

import os
from typing import Optional, Union

from transformers import AutoModelForCausalLM

from deepsparse import DEEPSPARSE_ENGINE, ORT_ENGINE, Pipeline


__all__ = ["text_generation_model_from_target", "get_save_path"]


def get_save_path(
    type_serialization: str,
    save_path: Optional[str] = None,
    default_file_name: str = "results",
) -> str:
    """
    Get the save path for the results file.

    :param type_serialization: The type of serialization
        to use for the results file.
    :param save_path: The path to save the results file to.
        If None, will save to the current directory.
    :param default_file_name: The default file name to use
        for the results file.
    :return: The save path for the results file.
    """
    file_name = default_file_name + "." + type_serialization
    if save_path is None:
        base_path = os.getcwd()
    else:
        base_path = save_path

    return os.path.join(base_path, file_name)


# TODO: Make it more generic to support sparsified models.
# TODO: Ideally import this functionality from SparseZoo.
def text_generation_model_from_target(
    target: str, engine_type: str, **kwargs
) -> Union[Pipeline, AutoModelForCausalLM]:
    """
    :param target: The target path to initialize the
        text generation model from. This can be a local
        or remote path to the model or a sparsezoo stub
    :param engine_type: The engine type to initialize the model with.
    :param kwargs: Additional kwargs to pass to the model initialization
    :return: The initialized model
    """
    if engine_type in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        return Pipeline.create(
            task="text-generation", model_path=target, engine_type=engine_type, **kwargs
        )
    try:
        # for now assume that if it's not a pipeline, it's a huggingface model
        return AutoModelForCausalLM.from_pretrained(target)
    except NotImplementedError as e:  # noqa: F841
        raise NotImplementedError(f"Unsupported engine type: {engine_type}")
