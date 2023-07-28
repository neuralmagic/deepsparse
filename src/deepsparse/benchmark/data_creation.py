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

import glob
import random
import string
from typing import Dict, List, Tuple

import numpy

from deepsparse import Pipeline


DEFAULT_STRING_LENGTH = 50
DEFAULT_IMAGE_SHAPE = (240, 240, 3)

__all__ = [
    "get_input_schema_type",
    "get_files_with_endings",
    "generate_sentence",
    "generate_image_data",
    "load_image_data",
    "generate_text_data",
    "load_text_data",
    "generate_question_data",
    "load_question_data",
]


def get_input_schema_type(pipeline: Pipeline) -> str:
    input_schema_requirements = list(pipeline.input_schema.__fields__.keys())
    input_schema_fields = pipeline.input_schema.__fields__

    if "images" in input_schema_requirements:
        return "image"
    if "sequences" in input_schema_requirements:
        sequence_types = [
            f.outer_type_ for f in input_schema_fields["sequences"].sub_fields
        ]
        if List[str] in sequence_types:
            return "text_sequence"
    elif "inputs" in input_schema_requirements:
        sequence_types = [
            f.outer_type_ for f in input_schema_fields["inputs"].sub_fields
        ]
        if List[str] in sequence_types:
            return "text_inputs"
    elif "question" in input_schema_requirements:
        return "question"

    raise Exception("Unknown schema requirement {}".format(input_schema_requirements))


def get_files_with_endings(
    folder: str, num_files: int, recursive: bool, file_endings: List[str]
) -> List[str]:
    files = []
    for f in glob.glob(folder + "/**", recursivere=recursive):
        if f.lower().endswith(file_endings):
            files.append(f)
    if len(files) < num_files:
        raise Exception("Not enough images found in {}".format(folder))
    return random.sample(files, num_files)


def generate_sentence(string_length: int, avg_word_length: int = 5):
    random_chars = "".join(random.choices(string.ascii_letters, k=string_length))
    space_locations = random.sample(
        range(string_length), int(string_length / avg_word_length)
    )
    random_chars = list(random_chars)
    for loc in space_locations:
        random_chars[loc] = " "
    return "".join(random_chars)


def generate_image_data(
    config: Dict, batch_size: int, logger: object
) -> List[numpy.ndarray]:
    input_data = []
    if "input_image_shape" in config and len(config["input_image_shape"]) == 3:
        image_shape = config["input_image_shape"]
    else:
        image_shape = DEFAULT_IMAGE_SHAPE
        logger.warning("Using default image shape {}".format(image_shape))

    for _ in range(batch_size):
        rand_array = numpy.random.randint(0, high=255, size=image_shape).astype(
            numpy.uint8
        )
        input_data.append(rand_array)

    return input_data


def load_image_data(config: Dict, batch_size: int) -> List[str]:
    path_to_data = config["data_folder"]
    recursive_search = config["recursive_search"]
    return get_files_with_endings(
        path_to_data, batch_size, recursive_search, [".jpg", ".jpeg", ".gif"]
    )


def generate_text_data(config: Dict, batch_size: int, logger: object) -> List[str]:
    input_data = []
    if "gen_sequence_length" in config:
        string_length = config["gen_sequence_length"]
    else:
        string_length = DEFAULT_STRING_LENGTH
        logger.warning("Using default string length {}".format(string_length))
    for _ in range(batch_size):
        rand_sentence = generate_sentence(string_length)
        input_data.append(rand_sentence)

    return input_data


def load_text_data(config: Dict, batch_size: int, logger: object) -> List[str]:
    path_to_data = config["data_folder"]
    recursive_search = config["recursive_search"]
    input_files = get_files_with_endings(
        path_to_data, batch_size, recursive_search, [".txt"]
    )
    if "max_string_length" in config:
        max_string_length = config["max_string_length"]
    else:
        max_string_length = -1
        logger.warning("Using default max string length {}".format(max_string_length))
    input_data = []
    for f_path in input_files:
        f = open(f_path)
        text_data = f.read()
        f.close()
        input_data.append(text_data[:max_string_length])
    return input_data


def generate_question_data(config: Dict, logger: object) -> Tuple[str, str]:
    if "gen_sequence_length" in config:
        string_length = config["gen_sequence_length"]
    else:
        string_length = DEFAULT_STRING_LENGTH
        logger.warning("Using default string length {}".format(string_length))
    question = generate_sentence(string_length)
    context = generate_sentence(string_length)
    return (question, context)


def load_question_data(config: Dict) -> Tuple[str, str]:
    path_to_questions = config["question_file"]
    path_to_context = config["context_file"]

    f_question = open(path_to_questions)
    f_context = open(path_to_context)
    question = f_question.read()
    context = f_context.read()
    f_question.close()
    f_context.close()
    return question, context
