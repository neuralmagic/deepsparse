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
import concurrent.futures
from pathlib import Path
import time
from typing import Optional

try:
    from fastapi import FastAPI
    import uvicorn
    from starlette.responses import RedirectResponse
except Exception as e:
    raise ImportError(
        "Server Dependencies not found, the recommended way of"
        "installing them is to use deepsparse[transformers,server]"
    )

from .schemas import (
    TaskRequestModel,
    TaskResponseModel,
)
from .throttled_engine import (
    get_throttled_pipeline,
    ThrottleWrapper,
    get_request_model,
    get_response_model,
)
from .utils import (
    parse_api_settings,
    fix_numpy_types,
)

app = FastAPI(
    title="DeepSparse-InferenceServer",
    version="0.1",
    description="DeepSparse Inference Server",
)

Engine: Optional[ThrottleWrapper] = None
RequestModel: Optional[TaskRequestModel] = None
ResponseModel: Optional[TaskResponseModel] = None


@app.get("/", include_in_schema=False)
def docs_redirect():
    """
    Redirect home to documentation
    """
    return RedirectResponse("/docs")


@app.post("/predict", response_model=ResponseModel,
          tags=["QA"])
def predict(qa_item: get_request_model(), ):
    """
    Process a pipeline task
    """
    _setup()
    results_future = Engine(
        **vars(qa_item)
    )
    return _resolve_future(results_future)


@fix_numpy_types
def _resolve_future(future: concurrent.futures.Future):
    # Wait for future to resolve and return result
    while not future.done():
        time.sleep(0.0001)
    result = future.result()
    return result


def _setup():
    # Initialize Engine, Request and Response Models
    # if not yet initialized
    global Engine, RequestModel, ResponseModel
    if not Engine:
        Engine = get_throttled_pipeline()
    if not RequestModel:
        RequestModel = get_request_model()
    if not ResponseModel:
        ResponseModel = get_response_model()


def main():
    """
    Driver Function
    """
    api_settings = parse_api_settings()
    app_name = f"{Path(__file__).stem}:app"
    _setup()
    uvicorn.run(app_name, host=api_settings.host, port=api_settings.port,
                log_level="info", workers=api_settings.workers)


if __name__ == '__main__':
    main()
