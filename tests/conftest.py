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
from subprocess import Popen
from typing import List

import pytest
from tests.helpers import delete_file


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
