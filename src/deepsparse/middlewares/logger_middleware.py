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

from enum import Enum
from typing import Any, Dict, Generator, List, Tuple

import numpy

from deepsparse.middlewares.middleware import MiddlewareCallable


NAME_KEY = "name"
SCALAR_TYPES_TUPLE = (
    int,
    float,
    bool,
    str,
)


class LoggerMiddleware(MiddlewareCallable):
    def __init__(
        self, call_next: MiddlewareCallable, identifier: str = "LoggerMiddleware"
    ):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:

        tag = kwargs.get(NAME_KEY)

        inference_state = kwargs.get("inference_state")
        if inference_state and hasattr(inference_state, "logger"):
            logger = inference_state.logger
            rtn = self.call_next(*args, **kwargs)

            run_time = None
            if hasattr(inference_state, "timer"):
                measurements = inference_state.timer.measurements.get(tag)
                if measurements:
                    # current run is the latest run
                    run_time = measurements[-1]

            for capture, value in unravel_value_as_generator(rtn, tag):
                logger.log(value=value, tag=tag, capture=capture, run_time=run_time)
            return rtn

        return self.call_next(*args, **kwargs)


def unravel_value_as_generator(
    value: Any, capture: str = ""
) -> Generator[Tuple[str, Any], None, None]:

    if isinstance(value, Dict):
        for key, val in value.items():
            new_capture = capture + f"['{key}']"
            yield from unravel_value_as_generator(val, new_capture)

    elif isinstance(value, numpy.ndarray):
        yield (capture, value)

    elif isinstance(value, Tuple) and not isinstance(value, SCALAR_TYPES_TUPLE):
        for idx, val in enumerate(value):
            new_capture = capture + f"[{idx}]"
            yield from unravel_value_as_generator(val, new_capture)

    elif isinstance(value, List):
        for idx, val in enumerate(value):
            new_capture = capture + f"[{idx}]"
            yield from unravel_value_as_generator(val, new_capture)

    elif isinstance(value, Enum):
        yield (capture.lstrip("."), value.value)

    elif isinstance(value, object) and not isinstance(value, SCALAR_TYPES_TUPLE):

        if hasattr(value, "__dict__"):
            for prop, val in vars(value).items():
                new_capture = capture + f".{prop}"
                yield from unravel_value_as_generator(val, new_capture)

        else:  # None type only
            yield (capture, None)

    else:
        # scalars: (int, float, bool, str)
        yield (capture, value)
