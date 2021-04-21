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
import pathlib


"""
Functionality for storing and setting the version info for DeepSparse.  If
a file named 'generated-version.py' exists, read version info from there, otherwise
fall back to defaults.
"""


__all__ = [
    "__version__",
    "version",
    "version_major",
    "version_minor",
    "version_bug",
    "version_build",
    "version_major_minor",
    "splash",
    "is_release",
]


# check for the backend's built version file, if it exists use that for version info
# otherwise, fall back to version info in this file
_deepsparse_dir = pathlib.Path(__file__).parent.absolute()
_gen_version_file = (
    os.path.join(_deepsparse_dir, "generated-version.py")
    if __file__ != "<input>"
    else os.path.join(_deepsparse_dir, "src", "deepsparse", "generated-version.py")
)  # <input> is a special case from exec on the file done in setup.py and docs build

if os.path.isfile(_gen_version_file):
    exec(open(_gen_version_file).read())
else:
    __version__ = "0.2.0"
    version = __version__
    splash = (
        "DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. "
        f"version: {version} (release)"
    )

is_release = len(version.split(".")) < 4  # build number not included for releases
version_major, version_minor, version_bug, version_build = version.split(".") + (
    [None] if is_release else []
)  # handle conditional for version being 3 parts or 4 (4 containing build date)
version_major_minor = f"{version_major}.{version_minor}"
