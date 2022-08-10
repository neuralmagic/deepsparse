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

import pytest
from deepsparse.pipeline import (
    _REGISTERED_PIPELINES,
    Pipeline,
    _dynamic_import_task,
    _module_to_dir_and_name,
)
from deepsparse.transformers.pipelines.question_answering import (
    QuestionAnsweringPipeline,
)


def test_module_dirs_and_names():
    assert _module_to_dir_and_name("a") == ("", "a")
    assert _module_to_dir_and_name("a.b") == ("a", "b")
    assert _module_to_dir_and_name("a.b.c") == ("a/b", "c")
    assert _module_to_dir_and_name("a.b.c.d") == ("a/b/c", "d")


def test_dynamic_import_raises_file_not_found():
    with pytest.raises(
        FileNotFoundError,
        match="Unable to find file for a.b.c. Looked for c.py under a/b",
    ):
        _dynamic_import_task("a.b.c")

    with pytest.raises(
        FileNotFoundError,
        match="Unable to find file for c. Looked for c.py under .",
    ):
        _dynamic_import_task("c")


def test_dynamic_import_no_task():
    with pytest.raises(
        RuntimeError,
        match="module must set the `TASK` attribute.",
    ):
        _dynamic_import_task(
            "tests.deepsparse.pipelines.dynamic_import_modules.no_task"
        )


def test_dynamic_import_no_register():
    with pytest.raises(
        RuntimeError,
        match="the file must register a pipeline",
    ):
        _dynamic_import_task(
            "tests.deepsparse.pipelines.dynamic_import_modules.no_register"
        )


def test_good_dynamic_import():
    assert "unit_test_task" not in _REGISTERED_PIPELINES
    _dynamic_import_task("tests.deepsparse.pipelines.dynamic_import_modules.ok")
    assert "unit_test_task" in _REGISTERED_PIPELINES


def test_pipeline_create_dynamic_task():
    constructor = Pipeline._get_task_constructor(
        "import:tests.deepsparse.pipelines.dynamic_import_modules.ok"
    )
    assert constructor == QuestionAnsweringPipeline
