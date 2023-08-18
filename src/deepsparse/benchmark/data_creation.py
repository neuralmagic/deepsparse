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
import logging
import random
import string
from os import path
from typing import Dict, List, Tuple

import numpy

from deepsparse import Pipeline
from deepsparse.benchmark.config import PipelineBenchmarkConfig


__all__ = [
    "get_input_schema_type",
    "generate_random_image_data",
    "load_image_data",
    "generate_random_text_data",
    "load_text_data",
    "generate_random_question_data",
    "load_question_data",
]

_LOGGER = logging.getLogger(__name__)

DEFAULT_STRING_LENGTH = 50
DEFAULT_IMAGE_SHAPE = (240, 240, 3)


class SchemaType:
    IMAGE: str = "images"
    TEXT_SEQ: str = "sequences"
    TEXT_INPUT: str = "inputs"
    QUESTION: str = "question"


def get_input_schema_type(pipeline: Pipeline) -> str:
    input_schema_requirements = list(pipeline.input_schema.__fields__.keys())
    input_schema_fields = pipeline.input_schema.__fields__

    if SchemaType.IMAGE in input_schema_requirements:
        return SchemaType.IMAGE
    if SchemaType.TEXT_SEQ in input_schema_requirements:
        sequence_types = [
            f.outer_type_ for f in input_schema_fields[SchemaType.TEXT_SEQ].sub_fields
        ]
        if List[str] in sequence_types:
            return SchemaType.TEXT_SEQ
    elif SchemaType.TEXT_INPUT in input_schema_requirements:
        sequence_types = [
            f.outer_type_ for f in input_schema_fields[SchemaType.TEXT_INPUT].sub_fields
        ]
        if List[str] in sequence_types:
            return SchemaType.TEXT_INPUT
    elif SchemaType.QUESTION in input_schema_requirements:
        return SchemaType.QUESTION

    raise Exception("Unknown schema requirement {}".format(input_schema_requirements))


def get_files_with_suffixes(
    folder: str, num_files: int, recursive: bool, file_endings: Tuple[str]
) -> List[str]:
    if not path.exists(folder):
        raise Exception("Can't parse files, {} does not exist".format(folder))
    files = []
    for f in glob.glob(folder + "/**", recursive=recursive):
        if f.lower().endswith(file_endings):
            files.append(f)
    if len(files) < num_files:
        raise Exception("Not enough images found in {}".format(folder))
    return random.sample(files, num_files)


def generate_random_sentence(string_length: int, avg_word_length: int = 5):
    random_chars = "".join(random.choices(string.ascii_letters, k=string_length))
    space_locations = random.sample(
        range(string_length), int(string_length / avg_word_length)
    )
    random_chars = list(random_chars)
    for loc in space_locations:
        random_chars[loc] = " "
    return "".join(random_chars)


def generate_random_image_data(
    config: PipelineBenchmarkConfig, batch_size: int
) -> List[numpy.ndarray]:
    input_data = []
    if config.input_image_shape and len(config.input_image_shape) == 3:
        image_shape = config.input_image_shape
    else:
        image_shape = DEFAULT_IMAGE_SHAPE
        _LOGGER.warning(
            f"Could not parse {config.input_image_shape}, "
            f"Using default image shape {image_shape}"
        )

    input_data = [
        numpy.random.randint(0, high=255, size=image_shape).astype(numpy.uint8)
        for _ in range(batch_size)
    ]
    return input_data


def load_image_data(config: PipelineBenchmarkConfig, batch_size: int) -> List[str]:
    if not config.data_folder:
        raise Exception("Data folder must be defined for real inputs")
    path_to_data = config.data_folder
    recursive_search = config.recursive_search
    return get_files_with_suffixes(
        path_to_data, batch_size, recursive_search, (".jpg", ".jpeg", ".gif")
    )


def generate_random_text_data(
    config: PipelineBenchmarkConfig, batch_size: int, avg_word_len=5
) -> List[str]:
    if config.gen_sequence_length:
        string_length = config.gen_sequence_length
    else:
        string_length = DEFAULT_STRING_LENGTH
        _LOGGER.warning("Ssing default string length %d" % string_length)

    input_data = [
        generate_random_sentence(string_length, avg_word_length=avg_word_len)
        for _ in range(batch_size)
    ]
    return input_data


def load_text_data(config: PipelineBenchmarkConfig, batch_size: int) -> List[str]:
    if not config.data_folder:
        raise Exception("Data folder must be defined for real inputs")
    path_to_data = config.data_folder
    recursive_search = config.recursive_search
    input_files = get_files_with_suffixes(
        path_to_data, batch_size, recursive_search, (".txt")
    )
    if config.max_string_length:
        max_string_length = config.max_string_length
    else:
        max_string_length = -1
        _LOGGER.warning("Using default max string length %d" % max_string_length)
    input_data = []
    for f_path in input_files:
        with open(f_path) as f:
            text_data = f.read()
            input_data.append(text_data[:max_string_length])
    return input_data


def generate_random_question_data(
    config: PipelineBenchmarkConfig, batch_size: int, avg_word_len=5
) -> Tuple[str, str]:
    if batch_size != 1:
        _LOGGER.warning(
            "Only batch size of 1 supported for Question Answering Pipeline"
        )
    if config.gen_sequence_length:
        string_length = config.gen_sequence_length
    else:
        string_length = DEFAULT_STRING_LENGTH
        _LOGGER.warning("Using default string length %d" % string_length)
    question = generate_random_sentence(string_length, avg_word_length=avg_word_len)
    context = generate_random_sentence(string_length, avg_word_length=avg_word_len)
    return (question, context)


def load_question_data(config: Dict, batch_size: int) -> Tuple[str, str]:
    if batch_size != 1:
        _LOGGER.warning(
            "Only batch size of 1 supported for Question Answering Pipeline"
        )
    if not config.question_file or not config.context_file:
        raise Exception(
            "Question and context files must be defined for question_answering pieline"
        )
    path_to_questions = config.question_file
    path_to_context = config.context_file

    question = ""
    context = ""
    with open(path_to_questions) as f:
        question = f.read()
    with open(path_to_context) as f:
        context = f.read()
    return question, context
