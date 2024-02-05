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

from deepsparse import Pipeline


__all__ = [
    "create_pipeline",
    "get_save_path",
    "args_to_dict",
    "resolve_integration",
]

LM_EVALUATION_HARNESS = "lm-evaluation-harness"


def potentially_check_dependency_import(integration_name: str) -> bool:
    """
    Check if the `integration_name` requires importing a dependency.
    If so, check if the dependency is installed and return True if it is.
    Otherwise, return False.

    :param integration_name: The name of the integration to check
    :return: True if the dependency is installed, False otherwise
    """

    if integration_name.replace("_", "-") == LM_EVALUATION_HARNESS:
        from deepsparse.evaluation.integrations import try_import_lm_evaluation_harness

        try_import_lm_evaluation_harness(raise_error=True)

    return True


def resolve_integration(
    pipeline: Pipeline, datasets: Union[str, List[str]]
) -> Union[str, None]:
    """
    Given a pipeline and dataset, infer the name of the evaluation integration
    to use. If unable to infer a name, return None.

    Currently:
        if the model is a generative language model,
        default to 'lm-evaluation-harness' otherwise return None

    :param pipeline: The pipeline to infer the integration for
    :param datasets: The datasets to infer the integration for
    :return: The name of the integration to use or None if unable to infer
    """
    if if_generative_language_model(pipeline):
        return LM_EVALUATION_HARNESS
    return None


def if_generative_language_model(pipeline: Pipeline) -> bool:
    """
    Checks if the model is a generative language model.
    """
    pipeline_name = pipeline.__class__.__name__
    if pipeline_name == "TextGenerationPipeline" or (
        pipeline_name == "TextGenerationPipelineNoKVCache"
    ):
        return True

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


def create_pipeline(
    model_path: str,
    engine_type: Optional[str] = None,
    **kwargs,
) -> Pipeline:
    """
    Create a pipeline for evaluation

    Note: This function is currently primarily
    focused on creating pipelines of type 'text-generation'
    This function will be expanded in the future to support
    more tasks and models

    :param model_path: The target path to initialize the
        text generation model from. This can be a local
        or remote path to the model or a sparsezoo stub
    :param engine_type: The engine type to initialize the model with.
    :return: The initialized pipeline
    """
    return Pipeline.create(
        task=kwargs.pop("task", "text-generation"),
        model_path=model_path,
        sequence_length=kwargs.pop("sequence_length", 2048),
        engine_type=engine_type,
        batch_size=kwargs.pop("batch_size", 1),
        **kwargs,
    )
