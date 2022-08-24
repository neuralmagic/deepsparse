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


import click

from deepsparse.server.server import start_server


@click.command()
@click.argument("config-path", type=str)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help=(
        "Bind socket to this host. Use --host 0.0.0.0 to make the application "
        "available on your local network. "
        "IPv6 addresses are supported, for example: --host '::'. Defaults to 0.0.0.0"
    ),
)
@click.option(
    "--port",
    type=int,
    default=5543,
    help="Bind to a socket with this port. Defaults to 5543.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warn", "critical", "fatal"], case_sensitive=False
    ),
    default="info",
    help="Sets the logging level. Defaults to info.",
)
def main(config_path: str, host: str, port: int, log_level: str):
    start_server(config_path, host, port, log_level)


if __name__ == "__main__":
    main()
