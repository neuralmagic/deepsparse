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
import sys
import argparse
import tarfile
from io import BytesIO
from urllib.request import Request, urlopen

__all__ = ["check_wand_binaries_exist", "download_wand_binaries"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models in the DeepSparse Engine"
    )

    parser.add_argument(
        "package_path",
        type=str,
        help="Path to the root of the local deepsparse package"
        "i.e. '/home/ubuntu/deepsparse/src/deepsparse'",
    )
    parser.add_argument(
        "--force_update",
        help="Force updating binaries without checking",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def check_wand_binaries_exist(package_path: str):
    """
    Check if the binaries neccessary to run the DeepSparse Engine are present
    """
    arch_path = os.path.join(package_path, "arch.bin")
    # TODO: Check for instruction set i.e.
    # avx2_path = os.path.join(package_path, "avx2", "deepsparse_engine.so")

    print("Checking to see if", arch_path, "exists..", os.path.exists(arch_path))
    return os.path.exists(arch_path)


def download_wand_binaries(
    package_path: str, full_version: str, is_release: bool = False
):
    """
    Pull down the binaries from the artifact store based on known version information
    and extract them to the right location
    """
    release_string = "release" if is_release else "nightly"

    print(
        f"Unable to find wand binaries locally in {package_path}.\n"
        f"Pulling down from artifact store for wand {release_string} {full_version}"
    )
    artifact_url = (
        "https://artifacts.neuralmagic.com/"
        f"{release_string}/"
        f"wand_nightly-{full_version}"
        f"-cp{sys.version_info[0]}{sys.version_info[1]}"
        f"-cp{sys.version_info[0]}{sys.version_info[1]}"
        f"{'' if sys.version_info[1] > 7 else 'm'}"  # 3.6 and 3.7 have a 'm'
        "-manylinux_x86_64.tar.gz"
    )

    print("Requesting", artifact_url)
    req = urlopen(Request(artifact_url, headers={"User-Agent": "Mozilla/5.0"}))
    tar = tarfile.open(name=None, fileobj=BytesIO(req.read()))
    # TODO: Base directory is included in the tarfile, so need to strip it to
    # extract files in the right place
    base_tar_dir = tar.getnames()[0]
    for member in tar.getmembers():
        # Skip root dir
        if member.name is base_tar_dir:
            continue
        # Remove base folder from each member
        member.name = member.name.replace(base_tar_dir + "/", "")
        tar.extract(member, package_path)


def main():
    args = parse_args()

    if args.force_update or not check_wand_binaries_exist(args.package_path):

        # default variables to be overwritten by the version.py file
        is_release = None
        version = "unknown"
        version_major = version
        version_minor = version
        version_bug = version
        # load and overwrite version and release info from deepsparse package
        version_path = os.path.join(args.package_path, "generated_version.py")
        if not os.path.exists(version_path):
            version_path = os.path.join(args.package_path, "version.py")
        locals_dict = {}
        exec(open(version_path).read(), globals(), locals_dict)
        is_release = locals_dict["is_release"]
        version = locals_dict["version"]
        version_major = locals_dict["version_major"]
        version_minor = locals_dict["version_minor"]
        version_bug = locals_dict["version_bug"]
        print(f"Loaded version {version} from {version_path}")
        full_version = f"{version_major}.{version_minor}.{version_bug}"

        download_wand_binaries(args.package_path, full_version, is_release)


if __name__ == "__main__":
    main()