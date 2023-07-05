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

import os
import tempfile
from subprocess import Popen
from typing import List
import logging
import shutil

import pytest
from tests.helpers import delete_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    torch_import_error = None
except err as torch_import_err:
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
    start_files_root = _get_files(directory=r".")
    start_files_temp = _get_files(directory=tempfile.gettempdir())
    yield
    end_files_root = _get_files(directory=r".")
    end_files_temp = _get_files(directory=tempfile.gettempdir())

    max_allowed_number_created_files = 4
    # GHA needs to create following files:
    # pyproject.toml, CONTRIBUTING.md, LICENSE, setup.cfg
    assert len(start_files_root) + max_allowed_number_created_files >= len(
        end_files_root
    ), (
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
    def get(return_jit: bool=False, **kwargs):
        # [TODO]: Make a model factory if needed
        torchvision_instance = torchvision_fixture
        if torchvision_instance:
            model = torchvision_instance.models.resnet50(kwargs)
            # if return_jit: 
            #     # return torch.jit.script(model)
            #     return torch.jit.trace(model, torch.rand(1, 3, 224, 224))
            if return_jit:
                # model.eval()
                return torch.jit.script(model)

            return model
    return get



# @pytest.fixture(scope='session')
# def delete_torch_models_after_torchscrript_tests(request):
#     cache_dir = os.path.expanduser("~/.cache/torch")
#     shutil.rmtree(cache_dir)

# @pytest.fixture(autouse=True, scope="module")
# def delete_cached_torch_models():
#     cache_dir = os.path.expanduser("~/.cache/torch")
#     shutil.rmtree(cache_dir)
    
    

        
@pytest.fixture(scope="function")
def torchscript_test_setup(torchvision_model_fixture):
    path = os.path.expanduser(os.path.join(TORCH_HUB, "hub", "checkpoints"))

    resnet50_nn_module = torchvision_model_fixture(pretrained=True, return_jit=False)
    expr = r"^resnet50-[0-9a-z]+\.pt[h]?$"
    resnet50_nn_module_path = find_file_with_pattern(path, expr)

    resnet50_jit = torchvision_model_fixture(pretrained=True, return_jit=True)
    resnet50_jit_path = resnet50_nn_module_path.replace(".pth", ".pt")
    torch.jit.save(resnet50_jit, resnet50_jit_path)

    yield {
        "jit_model": resnet50_jit,
        "jit_model_path": resnet50_jit_path,
    }

    cache_dir = os.path.expanduser("~/.cache/torch")
    shutil.rmtree(cache_dir)
    assert os.path.exists(cache_dir) is False


