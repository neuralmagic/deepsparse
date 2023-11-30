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
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, PreTrainedModel

from deepsparse import DEEPSPARSE_ENGINE, ORT_ENGINE, Pipeline


__all__ = [
    "create_model_from_target",
    "get_save_path",
    "args_to_dict",
    "resolve_integration",
]


def resolve_integration(
    model: Union[Pipeline, PreTrainedModel], datasets: Union[str, List[str]]
) -> Union[str, None]:
    """
    Given a model and dataset, inferr the name of the evaluation integration
    to use. If unable to infer a name, return None.

    Currently:
        if the model is llm-type model, default to 'llm-evaluation-harness'
        otherwise return None

    :param model: The model to infer the integration for
    :param datasets: The datasets to infer the integration for
    :return: The name of the integration to use or None if unable to infer
    """
    if is_model_llm(model):
        return "llm-evaluation-harness"
    return None


def is_model_llm(model: Any) -> bool:
    """
    Checks if the model is a Large Language Model (LLM) type model.
    """
    if isinstance(model, Pipeline):
        return model.__class__.__name__ == "TextGenerationPipeline"
    elif isinstance(model, PreTrainedModel):
        return "CausalLM" in model.__class__.__name__
    else:
        return False


def args_to_dict(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Convert a tuple of args to a dict of args.

    :param args: The args to convert. Should be a tuple of alternating
        arg names and arg values e.g.('--arg1', 1, 'arg2', 2, -arg3', 3).
        The names can optionally have a '-' or `--` in front of them.
    :return: The converted args as a dict.
    """
    if len(args) == 0:
        return {}
    # names are uneven indices, values are even indices
    args_names = args[0::2]
    args_values = args[1::2]
    # remove any '-' or '--' from the names
    args_names = [name.lstrip("-") for name in args_names]

    return dict(zip(args_names, args_values))


def get_save_path(
    type_serialization: str,
    save_path: Optional[str] = None,
    file_name: str = "results",
) -> str:
    """
    Get the save path for the results file.

    :param type_serialization: The type of serialization
        to use for the results file.
    :param save_path: The path to save the results file to.
        If None, will save to the current directory.
    :param file_name: The file name to use
        for the results file.
    :return: The save path for the results file.
    """
    file_name = file_name + "." + type_serialization
    if save_path is None:
        base_path = os.getcwd()
    else:
        base_path = save_path

    return os.path.join(base_path, file_name)


def create_model_from_target(
    target: str,
    engine_type: Optional[str] = None,
    **kwargs,
) -> Union[Pipeline, AutoModelForCausalLM]:
    """
    Create a model or a pipeline from a target path.

    Note: This function is currently limited to:
        - creating pipelines of type 'text-generation'
        - creating dense huggingface models of type 'AutoModelForCausalLM'
    This function will be expanded in the future to support more
    model types and frameworks.

    :param target: The target path to initialize the
        text generation model from. This can be a local
        or remote path to the model or a sparsezoo stub
    :param engine_type: The engine type to initialize the model with.
    :return: The initialized model
    """
    if engine_type in [DEEPSPARSE_ENGINE, ORT_ENGINE]:
        return Pipeline.create(
            task="text-generation",
            model_path=target,
            sequence_length=kwargs.pop("sequence_length", 2048),
            trust_remote_code=kwargs.pop("trust_remote_code", False),
            engine_type=engine_type,
            batch_size=kwargs.pop("batch_size", 1),
            **kwargs,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(target, **kwargs)
