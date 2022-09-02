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


@click.command(context_settings=dict(show_default=True))
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-e",
    "--endpoint",
    required=True,
    type=str,
    help="The name of the endpoint to tune.",
)
@click.option(
    "-c",
    "--concurrency",
    required=True,
    type=int,
    help="The expected amount of concurrent requests.",
)
@click.option(
    "-b",
    "--batch-size",
    required=True,
    type=int,
    help="The expected batch size of requests.",
)
@click.option(
    "-o",
    "--optimize",
    required=True,
    type=click.Choice(["latency", "throughput"]),
    multiple=True,
    help="The metrics to optimize.",
)
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON file that contains list of input data to send.",
)
@click.option(
    "-t",
    "--timelimit",
    type=float,
    default=10,
    help=(
        "Total time to spend tuning. "
        "Split evenly between the number of options to evaluate."
    ),
)
def tune(**kwargs):
    """
    Optimizes the given metrics by benchmarking various configuration in CONFIG_PATH.
    """
    print(kwargs)


if __name__ == "__main__":
    tune()
