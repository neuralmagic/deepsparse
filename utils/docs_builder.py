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

import argparse
import glob
import os
import re
import subprocess
from distutils.dir_util import copy_tree
from typing import List

from bs4 import BeautifulSoup
from packaging import version


def parse_args():
    """
    Setup and parse command line arguments for using the script
    """
    parser = argparse.ArgumentParser(
        description="Create and package documentation for the repository"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="the source directory to read the source for the docs from",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="the destination directory to put the built docs",
    )

    return parser.parse_args()


def create_docs(src: str, dest: str):
    """
    Run the sphinx command to create the docs from src into dest.

    :param src: the source directory for docs
    :type src: str
    :param dest: the destination directory for docs
    :type dest: str
    """
    print("running sphinx-multiversion")
    res = subprocess.run(["sphinx-multiversion", src, dest])

    if not res.returncode == 0:
        raise Exception(f"{res.stdout} {res.stderr}")

    print("completed sphinx build")


def package_docs(dest: str):
    """
    Run any extra packaging commands to prep the docs for release.
    Ex: copies the latest version to the root so if a version isn't specified will load.

    :param dest: the destination directory the docs were built in
    :type dest: str
    """
    print(f"packaging docs at {dest}")
    folders = _get_docs_folders(dest)
    print(f"found {len(folders)} docs folders from build")
    latest = _get_latest_folder(folders)
    print(f"found latest version `{latest}`, copying to {dest}")
    _copy_to_root(dest, latest)
    print(f"copied version {latest} to root as default")
    print("fixing root links")
    _fix_html_files_version_links(dest, folders)
    print("root links fixed")


def _get_docs_folders(dest: str) -> List[str]:
    folders = os.listdir(dest)

    return folders


def _get_latest_folder(folders: List[str]) -> str:
    versioned_folders = [
        (folder, version.parse(folder[1:]))
        for folder in folders
        if re.match(r"^v[0-9]+\.[0-9]+\.[0-9]+$", folder)
    ]
    versioned_folders.sort(key=lambda ver: ver[1])

    # get the latest version
    if versioned_folders:
        return versioned_folders[-1][0]

    # fall back on main if available as default
    if "main" in folders:
        return "main"

    # fall back on any other folder sorted
    folders.sort()
    return folders[-1]


def _copy_to_root(dest: str, latest: str):
    latest_path = os.path.join(dest, latest)
    copy_tree(latest_path, dest)


def _fix_html_files_version_links(dest: str, folders: List[str]):
    for file in glob.glob(os.path.join(dest, "**", "*.html"), recursive=True):
        relative = os.path.relpath(file, dest)
        parent = relative.split(os.sep)[0]

        if parent in folders:
            continue

        _fix_html_version_links(file)


def _fix_html_version_links(file_path: str):
    html = open(file_path).read()
    soup = BeautifulSoup(html, "html.parser")

    for anchor in soup.find("div", {"class": "rst-other-versions"}).find_all("a"):
        if anchor["href"].startswith("../"):
            anchor["href"] = anchor["href"][3:]

    with open(file_path, "wb") as file:
        file.write(soup.prettify("utf-8"))


def main():
    args = parse_args()
    create_docs(args.src, args.dest)
    package_docs(args.dest)


if __name__ == "__main__":
    main()
