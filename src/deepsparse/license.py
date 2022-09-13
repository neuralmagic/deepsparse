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

  TOKEN_OR_PATH raw token or path to a text file containing one

Options:
  --help  Show this message and exit.
"""


import logging
import os
import shutil
import sys
from tempfile import NamedTemporaryFile

import click

from deepsparse.lib import get_neuralmagic_binaries_dir, init_deepsparse_lib


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

    _validate_license(candidate_license_file_path)
    _LOGGER.info("DeepSparse license successfully validated")

    # copy candidate file to {LICENSE_FILE} in same directory as NM engine binaries
    license_file_path = os.path.join(get_neuralmagic_binaries_dir(), LICENSE_FILE)
    shutil.copy(candidate_license_file_path, license_file_path)
    _LOGGER.info(f"DeepSparse license file written to {license_file_path}")


def _validate_license(token):
    deepsparse_lib = init_deepsparse_lib()

    # nothing happens if token is valid
    # if token is invalid, deepsparse_lib will raise appropriate error response
    try:
        deepsparse_lib.validate_license(token)
    except RuntimeError:
        # deepsparse_lib handles error messaging, exit after message
        sys.exit(0)


@click.command()
@click.argument("token_or_path", type=str)
def main(token_or_path: str):
    """
    Validates a token for a DeepSparse enterprise license and creates
    a token file to be read for future use

    TOKEN_OR_PATH raw token or path to a text file containing one
    """
    add_deepsparse_license(token_or_path)


if __name__ == "__main__":
    main()
