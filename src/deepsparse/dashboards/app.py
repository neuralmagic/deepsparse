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

import random
import time

from prometheus_client import Histogram, start_http_server


def process_request(t):
    """A dummy function that takes some time."""
    time.sleep(t)


if __name__ == "__main__":
    # Start up the server to expose the metrics.
    start_http_server(8000)
    h = Histogram("request_latency_seconds", "Description of histogram")
    # Generate some requests.
    print("start_server")
    while True:
        _time = time.time()
        process_request(random.random())
        h.observe(time.time() - _time)
