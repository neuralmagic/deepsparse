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


from deepsparse.operators import Operator
from deepsparse.transformers.schemas.text_generation_schemas import (
    GenerationDefaults,
    TextGenerationInput,
)
from deepsparse.utils import InferenceState, PipelineState


__all__ = ["ParseTextGenerationInputs"]


class ParseTextGenerationInputs(Operator):
    def run(
        self,
        *args,
        inference_state: InferenceState,
        pipeline_state: PipelineState,
        **kwargs,
    ):
        """
        :param args: in line argument can only have 1, must either be
            a complete TextGenerationInput object or `sequences` for
            a TextGenerationInput
        :param kwargs: if a TextGenerationInput is not provided, then
            these kwargs will be used to instantiate one
        :return: parsed TextGenerationInput object
        """
        if "sequences" in kwargs and "prompt" not in kwargs:
            # support prompt and sequences interchangeably
            kwargs["prompt"] = kwargs["sequences"]

        if (
            args
            and not isinstance(args[0], TextGenerationInput)
            and "prompt" not in kwargs
            and "sequences" not in kwargs
        ):
            # assume first argument is "sequences" (prompt) by default
            kwargs["prompt"] = args[0]
            args = args[1:]

        if kwargs:
            generation_kwargs = kwargs.get("generation_kwargs", {})
            for k, v in kwargs.items():
                if not generation_kwargs.get(k) and hasattr(GenerationDefaults, k):
                    generation_kwargs[k] = v

            kwargs["generation_kwargs"] = generation_kwargs

        if args and isinstance(args[0], TextGenerationInput):
            return args[0]
        else:
            return kwargs
