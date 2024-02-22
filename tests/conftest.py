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

import logging
import os
import shutil
import tempfile
from subprocess import Popen
from typing import List

import pytest
from tests.helpers import delete_file
from tests.utils.helpers import find_file_with_pattern


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch

    torch_import_error = None
except Exception as torch_import_err:
    torch_import_error = torch_import_err
    torch = None


def _get_files(directory: str) -> List[str]:
    list_filepaths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            list_filepaths.append(os.path.join(os.path.abspath(root), file))
    return list_filepaths


@pytest.fixture
def cleanup():
    filenames: List[str] = []
    env_names: List[str] = []
    processes: List[Popen] = []

    yield {"files": filenames, "env_vars": env_names, "processes": processes}

    print("\nfixture:cleanup - cleanup up leftovers")

    # unset env vars
    if env_names:
        print(f"fixture:cleanup - removing env vars: {', '.join(env_names)}")
        for name in env_names:
            del os.environ[name]

    # delete files
    if filenames:
        print(f"fixture:cleanup - removing files: {', '.join(filenames)}")
        for fn in filenames:
            delete_file(fn)

    # terminate processes (test itself should do this, this is a backstop/catch-all)
    if processes:
        print(
            "fixture:cleanup - sending SIGTERM to PIDs "
            f"{', '.join(str(p.pid) for p in processes)}"
        )
        for proc in processes:
            proc.terminate()


@pytest.fixture(scope="session", autouse=True)
def check_for_created_files():
    start_files_root = [
        f_path for f_path in _get_files(directory=r".") if "__pycache__" not in f_path
    ]
    start_files_temp = _get_files(directory=tempfile.gettempdir())
    yield
    # allow creation of __pycache__ directories
    end_files_root = [
        f_path for f_path in _get_files(directory=r".") if "__pycache__" not in f_path
    ]
    end_files_temp = _get_files(directory=tempfile.gettempdir())

    # GHA needs to create following files:
    allowed_created_files = [
        "pyproject.toml",
        "CONTRIBUTING.md",
        "LICENSE",
        "setup.cfg",
        "prometheus_logs.prom",
    ]
    end_files_root = [
        f_path
        for f_path in end_files_root
        if os.path.basename(f_path) not in allowed_created_files
    ]
    start_files_root = [
        f_path
        for f_path in start_files_root
        if os.path.basename(f_path) not in allowed_created_files
    ]
    assert len(start_files_root) >= len(end_files_root), (
        f"{len(end_files_root) - len(start_files_root)} "
        f"files created in current working "
        f"directory during pytest run. "
        f"Created files: {set(end_files_root) - set(start_files_root)}"
    )
    max_allowed_sized_temp_files_megabytes = 150
    size_of_temp_files_bytes = sum(
        os.path.getsize(path) for path in set(end_files_temp) - set(start_files_temp)
    )
    size_of_temp_files_megabytes = size_of_temp_files_bytes / 1024 / 1024
    assert max_allowed_sized_temp_files_megabytes >= size_of_temp_files_megabytes, (
        f"{size_of_temp_files_megabytes} "
        f"megabytes of temp files created in temp directory during pytest run. "
        f"Created files: {set(end_files_temp) - set(start_files_temp)}"
    )


@pytest.fixture
def torchvision_fixture():
    try:
        import torchvision

        return torchvision
    except ImportError:
        logger.error("Failed to import torchvision")
        raise


@pytest.fixture(scope="function")
def torchvision_model_fixture(torchvision_fixture):
    def get(return_jit: bool = False, **kwargs):
        # [TODO]: Make a model factory if needed
        torchvision_instance = torchvision_fixture
        if torchvision_instance:
            model = torchvision_instance.models.resnet50(kwargs)

            if return_jit:
                return torch.jit.script(model)

            return model

    return get


@pytest.fixture(scope="function")
def torchscript_test_setup(torchvision_model_fixture):
    path = os.path.expanduser(os.path.join("~/.cache/torch", "hub", "checkpoints"))

    torchvision_model_fixture(pretrained=True, return_jit=False)
    expr = r"^resnet50-[0-9a-z]+\.pt[h]?$"
    resnet50_nn_module_path = find_file_with_pattern(path, expr)
    resnet50_nn_module = torchvision_model_fixture(pretrained=True)

    resnet50_jit = torchvision_model_fixture(pretrained=True, return_jit=True)
    resnet50_jit_path = resnet50_nn_module_path.replace(".pth", ".pt")
    torch.jit.save(resnet50_jit, resnet50_jit_path)

    yield {
        "jit_model": resnet50_jit,
        "jit_model_path": resnet50_jit_path,
        "nn_module_model": resnet50_nn_module,
    }

    cache_dir = os.path.expanduser("~/.cache/torch")
    shutil.rmtree(cache_dir)
    assert os.path.exists(cache_dir) is False
