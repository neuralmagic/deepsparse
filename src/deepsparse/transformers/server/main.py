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
DeepSparse Inference Server for tasks supported by deepsparse.transformers'
Pipeline module

##########
usage: Run DeepSparse Inference Server [-h] [--host HOST] [-p PORT]
                                       [-w WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --host HOST, -H HOST  The IP address of the hosted model
  -p PORT, --port PORT  The port that the model is hosted on
  -w WORKERS, --workers WORKERS
                        The number of workers to use for uvicorn

##########
Example:
1) From project root
$deepsparse.transformers.serve --model-path <MODEL-PATH>
"""
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Optional


try:
    import uvicorn
    from fastapi import FastAPI
    from starlette.responses import RedirectResponse
except Exception:
    raise ImportError(
        "Server Dependencies not found, the recommended way of"
        "installing them is to use deepsparse[server], transformers"
        "dependency is installed automatically if "
        "NM_NO_AUTOINSTALL_TRANSFORMERS env variable is not set."
    )

from deepsparse.transformers.helpers import fix_numpy_types

from .schemas import TaskRequestModel, TaskResponseModel
from .throttled_engine import (
    ThrottleWrapper,
    get_request_model,
    get_response_model,
    get_throttled_engine_pipeline,
)
from .utils import parse_api_settings


app = FastAPI(
    title="DeepSparse-InferenceServer",
    version="0.1",
    description="DeepSparse Inference Server",
)

ENGINE: Optional[ThrottleWrapper] = get_throttled_engine_pipeline()
TASK_REQUEST_MODEL: Optional[TaskRequestModel] = get_request_model()
TASK_RESPONSE_MODEL: Optional[TaskResponseModel] = get_response_model()


@app.get("/", include_in_schema=False)
def docs_redirect():
    """
    Redirect home to documentation
    """
    return RedirectResponse("/docs")


@app.post(
    "/predict",
    response_model=TASK_RESPONSE_MODEL,
)
def predict(
    input_item: TASK_REQUEST_MODEL,
):
    """
    Process a pipeline task
    """
    results_future = ENGINE(**vars(input_item))
    return _resolve_future(future=results_future)


@fix_numpy_types
def _resolve_future(future: Future):
    # Wait for future to resolve and return result
    while not future.done():
        time.sleep(0.0001)
    result = future.result()
    return result


def main():
    """
    Driver Function
    """
    api_settings = parse_api_settings()
    filename = Path(__file__).stem
    package = "deepsparse.transformers.server"
    app_name = f"{package}.{filename}:app"
    if ENGINE is None:
        raise ValueError("`TASK` env variable must be set")
    uvicorn.run(
        app_name,
        host=api_settings.host,
        port=api_settings.port,
        log_level="info",
        workers=api_settings.workers,
    )


if __name__ == "__main__":
    main()
