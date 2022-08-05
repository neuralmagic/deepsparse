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
import shutil
import time
from subprocess import PIPE, STDOUT, CompletedProcess, run
from typing import List

import requests

from sparsezoo import Model


def delete_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)


def is_server_up(url: str) -> bool:
    try:
        res = requests.get(url)
        return res.status_code is not None
    except Exception:
        return False


def wait_for_server(url: str, retries: int, interval: int = 1) -> bool:
    for _ in range(retries):
        is_up = is_server_up(url)
        if is_up:
            print(f"server up and ready in ~{_} seconds...")
            return True
        time.sleep(interval)
    return False


def run_command(command: List[str]) -> CompletedProcess:
    """
    Run given command with custom config and return the completed process.
    :param command: command to be executed (formatted as `subprocess.run` expects)
    :return: completed process as received from `subprocess.run`
    """
    return run(command, stdout=PIPE, stderr=STDOUT, check=False, encoding="utf-8")


def predownload_stub(stub: str, copy_framework_files: bool = False) -> Model:
    """
    Download a model based on SparseZoo stub and return the Model object. If
    `copy_framework_files` is True (default: False), also copy model’s config.json and
    tokenizer.json files from the framework subfolder (e.g. `pytorch`) up into the model
    root folder.

    :return: SparseZoo Model object of downloaded model
    """
    model = Model(stub)
    model_path = model.path
    if copy_framework_files:
        # required for `deepsparse.transformers.run_inference` on local model files
        config_path = model.deployment.default.get_file("config.json").path
        tokenizer_config_path = model.deployment.default.get_file("tokenizer.json").path
        shutil.copy(config_path, model_path)
        shutil.copy(tokenizer_config_path, model_path)

    return model
