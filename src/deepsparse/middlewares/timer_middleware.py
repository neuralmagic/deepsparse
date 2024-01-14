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

from typing import Any

from deepsparse.middlewares.middleware import MiddlewareCallable


IS_NESTED_KEY = "is_nested"
NAME_KEY = "name"
INFERENCE_STATE_KEY = "inference_state"


class TimerMiddleware(MiddlewareCallable):
    def __init__(
        self, call_next: MiddlewareCallable, identifier: str = "TimerMiddleware"
    ):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:
        name = kwargs.get(NAME_KEY)
        is_nested = kwargs.pop(IS_NESTED_KEY, False)

        inference_state = kwargs.get(INFERENCE_STATE_KEY)
        timer = inference_state.timer
        with timer.time(id=name, enabled=not is_nested):
            rtn = self.call_next(*args, **kwargs)
            xx = unwrap_logged_value(rtn)
            breakpoint()
            return rtn

def unwrap_logged_value(
    value: Any, parent_identifier: str = "", seperator: str = "__"
# ) -> Generator[Tuple[str, Any], None, None]:
):
    """
    Unwrap the `value`, given that it may be a nested
    data structure
    e.g.
    ```
    value = {"foo": {"alice": 1, "bob": 2},
             "bazz": 2},
    for identifier, value in unwrap_logged_value(value):
        -> yields:
            "foo__alice", 1
            "foo__bob", 2
            "bazz", 2 (no unwrapping)
    ```

    :param value: The value to possibly unwrap
    :param parent_identifier: The identifier that may be prepended to the
        child identifier retrieved from the nested dictionary
    :param seperator: The seperator to use when composing the parent and child
        identifiers
    :return: A generator that:
        - if `value` is a dictionary:
            continues to unwrap the dictionary...
        - if `value` is a BatchResult object:
            yields the `parent_identifier` and items in `value`
        - if `value` is not a dictionary or BatchResult object
            yields the `parent_identifier` and `value`
        Note: `parent_identifier` is composed by connecting the keys over
            the multiple levels of nesting with the seperator value is extracted
            from the nested dictionary (corresponding to the appropriate composed
            identifier)
    """
    breakpoint()
    if not isinstance(value, dict):
        yield parent_identifier, value
    else:
        for child_identifier, child_value in value.items():
            new_parent_identifier = (
                f"{parent_identifier}{seperator}{child_identifier}"
                if parent_identifier
                else child_identifier
            )
            if isinstance(child_value, BatchResult):
                for child_value_item in child_value:
                    yield new_parent_identifier, child_value_item
            elif isinstance(child_value, dict):
                yield from unwrap_logged_value(
                    child_value, new_parent_identifier, seperator
                )
            else:
                yield new_parent_identifier, child_value


class BatchResult(list):
    """
    Wrapper class for a list of values that
    are derived from a set of batch data
    """