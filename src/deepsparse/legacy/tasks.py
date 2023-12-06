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

"""
Classes and implementations for supported tasks in the DeepSparse pipeline and system
"""

import importlib
import logging
import os
import sys
from collections import namedtuple
from typing import Iterable, List, Optional, Tuple


_LOGGER = logging.getLogger(__name__)

__all__ = ["SupportedTasks", "AliasedTask"]


class AliasedTask:
    """
    A task that can have multiple aliases to match to.
    For example, question_answering which can alias to qa as well

    :param name: the name of the task such as question_answering or text_classification
    :param aliases: the aliases the task can go by in addition to the name such as
        qa, glue, sentiment_analysis, etc
    """

    def __init__(self, name: str, aliases: List[str]):
        self._name = name
        self._aliases = aliases

    @property
    def name(self) -> str:
        """
        :return: the name of the task such as question_answering
        """
        return self._name

    @property
    def aliases(self) -> List[str]:
        """
        :return: the aliases the task can go by such as qa, glue, sentiment_analysis
        """
        return self._aliases

    def matches(self, task: str) -> bool:
        """
        :param task: the name of the task to check whether the given instance matches.
            Checks the current name as well as any aliases.
            Everything is compared at lower case and "-" and whitespace
            are replaced with "_".
        :return: True if task does match the current instance, False otherwise
        """
        task = task.lower().replace("-", "_")

        # replace whitespace with "_"
        task = "_".join(task.split())

        return task == self.name or task in self.aliases


class SupportedTasks:
    """
    The supported tasks in the DeepSparse pipeline and system
    """

    nlp = namedtuple(
        "nlp",
        [
            "question_answering",
            "text_classification",
            "token_classification",
            "zero_shot_text_classification",
            "transformers_embedding_extraction",
        ],
    )(
        question_answering=AliasedTask("question_answering", ["qa"]),
        text_classification=AliasedTask(
            "text_classification", ["glue", "sentiment_analysis"]
        ),
        token_classification=AliasedTask("token_classification", ["ner"]),
        zero_shot_text_classification=AliasedTask("zero_shot_text_classification", []),
        transformers_embedding_extraction=AliasedTask(
            "transformers_embedding_extraction", []
        ),
    )

    chat = namedtuple("chat", ["chatbot", "chat"])(
        chatbot=AliasedTask("chatbot", []), chat=AliasedTask("chat", [])
    )
    text_generation = namedtuple(
        "text_generation", ["text_generation", "opt", "bloom"]
    )(
        text_generation=AliasedTask("text_generation", []),
        opt=AliasedTask("opt", []),
        bloom=AliasedTask("bloom", []),
    )
    code_generation = namedtuple("code_generation", ["code_generation", "codegen"])(
        code_generation=AliasedTask("code_generation", []),
        codegen=AliasedTask("codegen", []),
    )

    image_classification = namedtuple("image_classification", ["image_classification"])(
        image_classification=AliasedTask(
            "image_classification",
            ["image_classification"],
        ),
    )

    yolo = namedtuple("yolo", ["yolo"])(
        yolo=AliasedTask("yolo", ["yolo"]),
    )
    yolov8 = namedtuple("yolov8", ["yolov8"])(
        yolov8=AliasedTask("yolov8", ["yolov8"]),
    )
    yolact = namedtuple("yolact", ["yolact"])(
        yolact=AliasedTask("yolact", ["yolact"]),
    )

    haystack = namedtuple("haystack", ["information_retrieval_haystack"])(
        information_retrieval_haystack=AliasedTask(
            "information_retrieval_haystack", ["haystack"]
        ),
    )
    embedding_extraction = namedtuple("embedding_extraction", ["embedding_extraction"])(
        embedding_extraction=AliasedTask(
            "embedding_extraction", ["embedding_extraction"]
        ),
    )
    open_pif_paf = namedtuple("open_pif_paf", ["open_pif_paf"])(
        open_pif_paf=AliasedTask("open_pif_paf", ["open_pif_paf"]),
    )

    all_task_categories = [
        nlp,
        image_classification,
        yolo,
        yolov8,
        yolact,
        haystack,
        embedding_extraction,
        open_pif_paf,
        text_generation,
        chat,
        code_generation,
    ]

    @classmethod
    def check_register_task(
        cls, task: str, extra_tasks: Optional[Iterable[str]] = None
    ):
        """
        :param task: task name to validate and import dependencies for
        :param extra_tasks: valid task names that are not included in supported tasks.
            i.e. tasks registered to Pipeline at runtime
        """
        if task == "custom":
            # custom task, register the CustomPipeline
            import deepsparse.pipelines.custom_pipeline  # noqa: F401

        elif cls.is_text_generation(task):
            import deepsparse.transformers.pipelines.text_generation  # noqa: F401

        elif cls.is_chat(task):
            import deepsparse.transformers.pipelines.chat  # noqa: F401

        elif cls.is_code_generation(task):
            import deepsparse.transformers.pipelines.code_generation  # noqa: F401

        elif cls.is_nlp(task):
            # trigger transformers pipelines to register with Pipeline.register
            import deepsparse.transformers.pipelines  # noqa: F401

        elif cls.is_image_classification(task):
            # trigger image classification pipelines to
            # register with Pipeline.register
            import deepsparse.image_classification.pipelines  # noqa: F401

        elif cls.is_yolact(task):
            # trigger yolo pipelines to register with Pipeline.register
            import deepsparse.yolact.pipelines  # noqa: F401

        elif cls.is_yolo(task):
            # trigger yolo pipelines to register with Pipeline.register
            import deepsparse.yolo.pipelines  # noqa: F401

        elif cls.is_yolov8(task):
            # trigger yolo pipelines to register with Pipeline.register
            import deepsparse.yolov8.pipelines  # noqa: F401

        elif cls.is_haystack(task):
            # trigger haystack pipeline as well as transformers pipelines to
            # register with Pipeline.register
            import deepsparse.transformers.haystack  # noqa: F401

        elif cls.is_embedding_extraction(task):
            # trigger embedding_extraction pipelines to register with
            #  Pipeline.register
            import deepsparse.pipelines.embedding_extraction  # noqa :F401

        elif cls.is_open_pif_paf(task):
            # trigger embedding_extraction pipelines to register with
            #  Pipeline.register
            import deepsparse.open_pif_paf.pipelines  # noqa :F401

        all_tasks = set(cls.task_names() + (list(extra_tasks or [])))
        if task not in all_tasks:
            raise ValueError(
                f"Unknown Pipeline task {task}. Currently supported tasks are "
                f"{list(all_tasks)}"
            )

    @classmethod
    def is_chat(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is a chat task
        :return: True if it is a chat task, False otherwise
        """
        return any(chat_task.matches(task) for chat_task in cls.chat)

    @classmethod
    def is_text_generation(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is a text generation task
            such as codegen
        :return: True if it is a text generation task, False otherwise
        """
        return any(
            text_generation_task.matches(task)
            for text_generation_task in cls.text_generation
        )

    @classmethod
    def is_code_generation(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is a text generation task
            such as codegen
        :return: True if it is a text generation task, False otherwise
        """
        return any(
            code_generation_task.matches(task)
            for code_generation_task in cls.code_generation
        )

    @classmethod
    def is_nlp(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is an nlp task
            such as question_answering
        :return: True if it is an nlp task, False otherwise
        """
        return any([nlp_task.matches(task) for nlp_task in cls.nlp])

    @classmethod
    def is_cv(cls, task: str) -> bool:
        return (
            cls.is_yolo(task)
            or cls.is_yolov8(task)
            or cls.is_yolact(task)
            or cls.is_image_classification(task)
            or cls.is_open_pif_paf(task)
        )

    @classmethod
    def is_image_classification(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is an image
            classification task
        :return: True if it is an image classification task, False otherwise
        """
        return any([ic_task.matches(task) for ic_task in cls.image_classification])

    @classmethod
    def is_yolo(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is an image
            segmentation task using YOLO
        :return: True if it is an segmentation task using YOLO, False otherwise
        """
        return any([yolo_task.matches(task) for yolo_task in cls.yolo])

    @classmethod
    def is_yolov8(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is an image
            segmentation task using YOLOv8
        :return: True if it is an segmentation task using YOLOv8, False otherwise
        """
        return any([yolov8_task.matches(task) for yolov8_task in cls.yolov8])

    @classmethod
    def is_yolact(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is an image
            segmentation task using YOLO
        :return: True if it is an segmentation task using YOLO, False otherwise
        """
        return any([yolact_task.matches(task) for yolact_task in cls.yolact])

    @classmethod
    def is_haystack(cls, task: str) -> bool:
        """
        :param task: the name of the task to check whether it is a haystack task
        :return: True if it is a haystack task, False otherwise
        """
        return any([haystack_task.matches(task) for haystack_task in cls.haystack])

    @classmethod
    def is_embedding_extraction(cls, task):
        """
        :param task: the name of the task to check whether it is an
            embedding_extraction task
        :return: True if it is an embedding_extraction task, False otherwise
        """
        return any(
            embedding_extraction_task.matches(task)
            for embedding_extraction_task in cls.embedding_extraction
        )

    @classmethod
    def is_open_pif_paf(cls, task):
        """
        :param task: the name of the task to check whether it is an
            embedding_extraction task
        :return: True if it is an open_pif_paf task, False otherwise
        """
        return any(
            open_pif_paf_task.matches(task) for open_pif_paf_task in cls.open_pif_paf
        )

    @classmethod
    def task_names(cls):
        task_names = ["custom"]
        for task_category in cls.all_task_categories:
            for task in task_category:
                unique_aliases = (
                    alias for alias in task._aliases if alias != task._name
                )
                task_names += (task._name, *unique_aliases)
        return task_names


def dynamic_import_task(module_or_path: str) -> str:
    """
    Dynamically imports `module` with importlib, and returns the `TASK`
    attribute on the module (something like `importlib.import_module(module).TASK`).

    Example contents of `module`:
    ```python
    from deepsparse.pipeline import Pipeline
    from deepsparse.transformers.pipelines.question_answering import (
        QuestionAnsweringPipeline,
    )

    TASK = "my_qa_task"
    Pipeline.register(TASK)(QuestionAnsweringPipeline)
    ```

    NOTE: this modifies `sys.path`.

    :raises FileNotFoundError: if path does not exist
    :raises RuntimeError: if the imported module does not contain `TASK`
    :raises RuntimeError: if the module doesn't register the task
    :return: The task from the imported module.
    """
    parent_dir, module_name = _split_dir_and_name(module_or_path)
    if not os.path.exists(os.path.join(parent_dir, module_name + ".py")):
        raise FileNotFoundError(
            f"Unable to find file for {module_or_path}. "
            f"Looked for {module_name}.py under {parent_dir if parent_dir else '.'}"
        )

    # add parent_dir to sys.path so we can import the file as a module
    sys.path.append(os.curdir)
    if parent_dir:
        _LOGGER.info(f"Adding {parent_dir} to sys.path")
        sys.path.append(parent_dir)

    # do the import
    _LOGGER.info(f"Importing '{module_name}'")
    module_or_path = importlib.import_module(module_name)

    if not hasattr(module_or_path, "TASK"):
        raise RuntimeError(
            "When using --task import:<module>, "
            "module must set the `TASK` attribute."
        )

    task = getattr(module_or_path, "TASK")
    _LOGGER.info(f"Using task={repr(task)}")

    return task


def _split_dir_and_name(module_or_path: str) -> Tuple[str, str]:
    """
    Examples:
    - `a` -> `("", "a")`
    - `a.b` -> `("a", "b")`
    - `a.b.c` -> `("a/b", "c")`

    :return: module split into directory & name
    """
    if module_or_path.endswith(".py"):
        # assume path
        split_char = os.sep
        module_or_path = module_or_path.replace(".py", "")
    else:
        # assume module
        split_char = "."
    *dirs, module_name = module_or_path.split(split_char)
    parent_dir = os.sep if dirs == [""] else os.sep.join(dirs)
    return parent_dir, module_name
