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
Functionality for storing and setting the version info for DeepSparse.  If
a file named 'generated_version.py' exists, read version info from there, otherwise
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
    "is_enterprise",
]


try:
    # check for the backend's built version file, if it exists use that for version info
    from deepsparse.generated_version import is_enterprise, is_release, splash, version
except Exception:
    # otherwise, fall back to version info in this file
    version = "1.3.0"
    is_release = False
    is_enterprise = False
    splash = (
        "DeepSparse Engine, Copyright 2021-present / Neuralmagic, Inc. "
        f"version: {version} COMMUNITY EDITION (release)"
    )

__version__ = version
version_major, version_minor, version_bug, version_build = version.split(".") + (
    [None] if len(version.split(".")) < 4 else []
)  # handle conditional for version being 3 parts or 4 (4 containing build date)
version_major_minor = f"{version_major}.{version_minor}"
