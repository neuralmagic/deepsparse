import os
import shutil
import time
from subprocess import PIPE, STDOUT, CompletedProcess, run
from typing import List

import requests

from sparsezoo import Model, Zoo


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
    model = Zoo.download_model_from_stub(stub)
    if copy_framework_files:
        # required for `deepsparse.transformers.run_inference` on local model files
        model_path = model.dir_path
        framework_path = os.path.join(model_path, model.framework)
        shutil.copy(
            os.path.join(framework_path, "config.json"), os.path.join(model_path)
        )
        shutil.copy(
            os.path.join(framework_path, "tokenizer.json"), os.path.join(model_path)
        )
    return model
