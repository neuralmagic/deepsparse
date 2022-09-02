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

import click

from deepsparse.lib import get_neuralmagic_binaries_dir


LICENSE_FILE = "license.txt"

_LOGGER = logging.getLogger(__name__)


def add_deepsparse_license(token_or_path):
    token = token_or_path
    if os.path.exists(token_or_path):
        with open(token_or_path) as token_file:
            token = token_file.read()

    _validate_token(token)

    # write to {LICENSE_FILE} in same directory as NM engine binaries
    license_file_path = os.path.join(get_neuralmagic_binaries_dir(), LICENSE_FILE)
    with open(license_file_path, "w") as license_file:
        license_file.write(token)

    _LOGGER.info(f"DeepSparse license file written to {license_file_path}")


def _validate_token(token):
    if not token:  # TODO: propagate validation from engine
        raise ValueError(
            "Sorry, it does not seem as if you have an active "
            "enterprise license for DeepSparse. If you believe "
            "this is a mistake, please reach out to license@neuralmagic.com "
            "for help."
        )


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
