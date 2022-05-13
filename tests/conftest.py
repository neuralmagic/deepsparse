import os
from subprocess import Popen
from typing import List

import pytest

from helpers import delete_file


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
            f"fixture:cleanup - sending SIGTERM to PIDs {', '.join(str(p.pid) for p in processes)}"
        )
        for proc in processes:
            proc.terminate()
