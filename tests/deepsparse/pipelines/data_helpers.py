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

from pathlib import Path
from typing import Any, Dict, List, Union


__all__ = ["create_test_inputs"]


def text_classification(
    batch_size: int = 1,
) -> Dict[str, Union[List[List[str]], List[str], str]]:
    """
    Create and return a test input for text classification

    :param batch_size: The batch_size of inputs to return
    :return: A dict representing inputs for text classification
    """
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), "batch size must be greater than 1"

    inputs_db = {1: "Dogs are nice", 0: "I am Batman"}
    inputs = [inputs_db[_ % 2] for _ in range(batch_size)]
    return {"sequences": inputs}


def token_classification(
    batch_size: int = 1,
) -> Dict[str, Union[Union[List[str], str]]]:
    """
    Create and return a test input for token classification

    :param batch_size: The batch_size of inputs to return
    :return: A dict representing inputs for token classification
    """
    inputs_db = {
        1: "U.N. official Ekeus heads for Baghdad",
        0: "WHO stands for World Health Organisation",
    }

    inputs = [inputs_db[_ % 2] for _ in range(batch_size)]
    return {"inputs": inputs}


def computer_vision(
    batch_size: int = 1,
) -> Dict[str, Union[Union[List[str], str, List[Any]]]]:
    """
    Create and return a test input with images

    :param batch_size: The batch_size of inputs to return
    :return: A dict representing inputs for CV based tasks
    """

    image_dir = Path(__file__).parents[0] / "sample_images"

    basilica_img_rel_path = image_dir / "basilica.jpg"
    buddy_img_rel_path = image_dir / "buddy.jpeg"

    basilica_image_abs_path = str(basilica_img_rel_path.absolute())
    buddy_image_abs_path = str(buddy_img_rel_path.absolute())

    inputs_db = {1: basilica_image_abs_path, 0: buddy_image_abs_path}

    images = [inputs_db[_ % 2] for _ in range(batch_size)]
    return {"images": images}


def create_test_inputs(task, batch_size: int = 1):
    """
    A function to generate test_inputs for a given task and batch_size

    :param task: The Pipeline task to generate inputs for
    :param batch_size: batch size of the inputs
    :return: A dict representing inputs for the task
    """
    dispatcher = {
        "text_classification": text_classification,
        "token_classification": token_classification,
        "yolo": computer_vision,
        "image_classification": computer_vision,
    }
    return dispatcher[task](batch_size=batch_size)
