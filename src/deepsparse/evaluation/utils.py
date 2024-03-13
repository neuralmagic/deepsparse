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
import ast
import logging
import os
from typing import Dict, List, Optional, Union

from deepsparse import Pipeline
from deepsparse.operators.engine_operator import DEEPSPARSE_ENGINE
from sparsezoo.utils.registry import standardize_lookup_name


__all__ = [
    "create_pipeline",
    "get_save_path",
    "parse_kwarg_tuples",
    "resolve_integration",
]
_LOGGER = logging.getLogger(__name__)

LM_EVALUATION_HARNESS = "lm-evaluation-harness"
LM_EVALUATION_HARNESS_ALIASES = ["lm-eval-harness", "lm-eval"]
PERPLEXITY = "perplexity"


def potentially_check_dependency_import(integration_name: str) -> bool:
    """
    Check if the `integration_name` requires importing a dependency.
    Checking involves comparing the `integration_name` to the known
    integrations (e.g. 'lm-evaluation-harness') or their aliases.
    If so, check if the dependency is installed and return True if it is.
    Otherwise, return False.

    :param integration_name: The name of the integration to check. The name
        is standardized using `standardize_lookup_name` before checking.
    :return: True if the dependency is installed, False otherwise
    """
    integration_name = standardize_lookup_name(integration_name)

    if integration_name == LM_EVALUATION_HARNESS or any(
        integration_name == alias for alias in LM_EVALUATION_HARNESS_ALIASES
    ):
        from deepsparse.evaluation.integrations import try_import_lm_evaluation_harness

        try_import_lm_evaluation_harness()
    if integration_name == PERPLEXITY:
        from deepsparse.evaluation.integrations.perplexity import (  # noqa F401
            integration_eval,
        )

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


def parse_kwarg_tuples(kwargs: tuple) -> Dict:
    """
    Convert a tuple of kwargs to a dict of kwargs.
    This function is used to enable the click parsing of kwargs.

    Example use:
    ```
    @click.command(
    context_settings=dict(
        ignore_unknown_options=True)
    )
    @click.argument(...)
    @click.option(...)
    ...
    @click.argument("kwargs", nargs=-1, type=click.UNPROCESSED)
    def main(..., kwargs):
        ...
        kwargs: Dict[str, Any] = parse_kwarg_tuples(kwargs: Tuple)
    ```

    Example inputs, outputs:
    ```
    input = ('--arg1', 1, 'arg2', 2, '-arg3', 3)
    output = parse_kwarg_tuples(input)
    output = {'arg1': 1, 'arg2': 2, 'arg3': 3}
    ```

    :param kwargs: The kwargs to convert. Should be a tuple of alternating
        kwargs names and kwargs values e.g.('--arg1', 1, 'arg2', 2, -arg3', 3).
        The names can optionally have a '-' or `--` in front of them.
    :return: The converted kwargs as a dict.
    """
    if len(kwargs) == 0:
        return {}
    if len(kwargs) % 2 != 0:
        raise ValueError(
            "kwargs must be a tuple of alternating names and values "
            "i.e. the length of kwargs tuple must be even. Received "
            f"kwargs: {kwargs}"
        )
    # names are uneven indices, values are even indices
    kwargs_names = kwargs[0::2]
    kwargs_values = kwargs[1::2]
    # by default kwargs values are strings, so convert them
    # to the appropriate type if possible
    kwargs_values = list(kwargs_values)
    for i, value in enumerate(kwargs_values):
        try:
            kwargs_values[i] = ast.literal_eval(value)
        except Exception as e:  # noqa E841
            _LOGGER.debug(
                f"Failed to infer non-string type"
                f"from kwarg value: {value}. It will"
                f"be left as a string."
            )

    # remove any '-' or '--' from the names
    kwargs_names = [name.lstrip("-") for name in kwargs_names]

    return dict(zip(kwargs_names, kwargs_values))


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
    :return: The initialized pipeline and the mutated
        (potentially reduced number of) kwargs
    """
    engine_type = engine_type or DEEPSPARSE_ENGINE
    return (
        Pipeline.create(
            task=kwargs.pop("task", "text-generation"),
            model_path=model_path,
            sequence_length=kwargs.pop("sequence_length", 2048),
            engine_type=engine_type,
            batch_size=kwargs.pop("batch_size", 1),
        ),
        kwargs,
    )
