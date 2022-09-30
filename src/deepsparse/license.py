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
Usage: deepsparse.license [OPTIONS] TOKEN_OR_PATH

  Validates a token for a DeepSparse enterprise license and creates a token
  file to be read for future use

  Token file by default will be written to license.txt in
  ~/.config/neuralmagic. This directory may be overwritten by setting the
  NM_CONFIG_DIR environment variable - this variable should then be set as
  well in any subsequent uses of the deepsparse engine

  TOKEN_OR_PATH raw token or path to a text file containing one

Options:
  --help  Show this message and exit.
"""


import logging
import os
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import click

from deepsparse.lib import init_deepsparse_lib


NM_CONFIG_DIR = "NM_CONFIG_DIR"
DEFAULT_CONFIG_DIR = os.path.join(Path.home(), ".config", "neuralmagic")
LICENSE_FILE = "license.txt"

_LOGGER = logging.getLogger(__name__)


def add_deepsparse_license(token_or_path):
    candidate_license_file_path = token_or_path
    if not os.path.exists(token_or_path):
        # write raw token to temp file for validadation
        candidate_license_tempfile = NamedTemporaryFile()
        candidate_license_file_path = candidate_license_tempfile.name
        with open(candidate_license_file_path, "w") as token_file:
            token_file.write(token_or_path)

    validate_license(candidate_license_file_path)
    _LOGGER.info("DeepSparse license successfully validated")

    # copy candidate file to {LICENSE_FILE} in same directory as NM engine binaries
    license_file_path = _get_license_file_path()
    shutil.copy(candidate_license_file_path, license_file_path)
    _LOGGER.info(f"DeepSparse license file written to {license_file_path}")
    validate_license()


def validate_license(license_path: Optional[str] = None):
    """
    Validates a candidate license token (JWT). Should be passed
    as a text file containing only the JWT. If no path is provided
    the expected file path of the token will be validated. Default
    path is ~/.config/neuralmagic/license.txt

    :param license_path: file path to text file of token to validate.
        Default is None, expected token path will be validated
    """

    deepsparse_lib = init_deepsparse_lib()

    # if token is invalid, deepsparse_lib will raise appropriate error response
    try:
        if license_path is None:
            splash_message = deepsparse_lib.validate_license()
            print(splash_message)
            return
        deepsparse_lib.validate_license(license_path)
    except RuntimeError:
        # deepsparse_lib handles error messaging, exit after message
        sys.exit(1)


def _get_license_file_path():
    # license file written to NM_CONFIG_DIR env var under license.txt
    # defaults to ~/.config/neuralmagic
    config_dir = os.environ.get(NM_CONFIG_DIR, DEFAULT_CONFIG_DIR)
    os.makedirs(config_dir, exist_ok=True)

    return os.path.join(config_dir, LICENSE_FILE)


@click.command()
@click.option("--license_path", type=str, default=None)
def validate_license_cli(license_path: Optional[str] = None):
    """
    Validates a candidate license token (JWT). Should be passed
    as a text file containing only the JWT. If no path is provided
    the expected file path of the token will be validated. Default
    path is ~/.config/neuralmagic/license.txt

    :param license_path: file path to text file of token to validate.
        Default is None, expected token path will be validated
    """
    validate_license(license_path)


@click.command()
@click.argument("token_or_path", type=str)
def main(token_or_path: str):
    """
    Validates a token for a DeepSparse enterprise license and creates
    a token file to be read for future use

    Token file by default will be written to license.txt in
    ~/.config/neuralmagic. This directory may be overwritten by setting
    the NM_CONFIG_DIR environment variable - this variable should then
    be set as well in any subsequent uses of the deepsparse engine

    TOKEN_OR_PATH raw token or path to a text file containing one
    """
    add_deepsparse_license(token_or_path)


if __name__ == "__main__":
    main()
