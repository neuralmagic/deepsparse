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
import time
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Optional, AsyncGenerator

import yaml

import uvicorn
from deepsparse.engine import Context
from deepsparse.loggers import BaseLogger
from deepsparse.pipeline import Pipeline
from deepsparse.server.config import (
    INTEGRATION_LOCAL,
    INTEGRATION_SAGEMAKER,
    INTEGRATION_OPENAI,
    INTEGRATIONS,
    EndpointConfig,
    ServerConfig,
    SystemLoggingConfig,
)
from deepsparse.server.config_hot_reloading import start_config_watcher
from deepsparse.server.helpers import (
    prep_outputs_for_serialization,
    server_logger_from_config,
)
from deepsparse.server.system_logging import (
    SystemLoggingMiddleware,
    log_system_information,
)
from fastapi import FastAPI, UploadFile, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import RedirectResponse

from deepsparse.server.outputs import CompletionOutput, RequestOutput
from deepsparse.server.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
    random_uuid
)
from deepsparse.server.openai_server import create_logprobs, create_error_response, HTTPStatus

_LOGGER = logging.getLogger(__name__)

def start_server(
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 5543,
    log_level: str = "info",
    hot_reload_config: bool = False,
):
    """
    Starts a FastAPI server with uvicorn with the configuration specified.

    :param config_path: A yaml file with the server config. See :class:`ServerConfig`.
    :param host: The IP address to bind the server to.
    :param port: The port to listen on.
    :param log_level: Log level given to python and uvicorn logging modules.
    :param hot_reload_config: `True` to reload the config file if it is modified.
    """
    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["loggers"][__name__] = {
        "handlers": ["default"],
        "level": log_level.upper(),
    }

    _LOGGER.info(f"config_path: {config_path}")

    with open(config_path) as fp:
        obj = yaml.safe_load(fp)
    server_config = ServerConfig(**obj)
    _LOGGER.info(f"Using config: {repr(server_config)}")

    if hot_reload_config:
        _LOGGER.info(f"Watching {config_path} for changes.")
        _ = start_config_watcher(config_path, f"http://{host}:{port}/endpoints", 0.5)


    if server_config.integration == INTEGRATION_OPENAI:
        from deepsparse.server.openai_server import SUPPORTED_TASKS

        for endpoint in server_config.endpoints:
            if endpoint.task not in SUPPORTED_TASKS:
                raise ValueError(f"The OpenAI server currently only supports {SUPPORTED_TASKS} tasks")

    app = _build_app(server_config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        log_config=log_config,
        # NOTE: only want to have 1 server process so models aren't being copied
        workers=1,
    )


def _build_app(server_config: ServerConfig) -> FastAPI:
    route_counts = Counter([cfg.route for cfg in server_config.endpoints])
    if route_counts[None] > 1:
        raise ValueError(
            "You must specify `route` for all endpoints if multiple endpoints are used."
        )

    for route, count in route_counts.items():
        if count > 1:
            raise ValueError(
                f"{route} specified {count} times for multiple EndpoingConfig.route"
            )

    if server_config.integration not in INTEGRATIONS:
        raise ValueError(
            f"Unknown integration field {server_config.integration}. "
            f"Expected one of {INTEGRATIONS}"
        )

    #_set_pytorch_num_threads(server_config)
    #_set_thread_pinning(server_config)

    context = Context(
        num_cores=server_config.num_cores,
        num_streams=server_config.num_workers,
    )
    executor = ThreadPoolExecutor(max_workers=context.num_streams)

    _LOGGER.info(f"Built context: {repr(context)}")
    _LOGGER.info(f"Built ThreadPoolExecutor with {executor._max_workers} workers")

    server_logger = server_logger_from_config(server_config)
    app = FastAPI()
    app.add_middleware(
        SystemLoggingMiddleware,
        server_logger=server_logger,
        system_logging_config=server_config.system_logging,
    )

    @app.get("/", include_in_schema=False)
    def _home():
        return RedirectResponse("/docs")

    @app.get("/config", tags=["general"], response_model=ServerConfig)
    def _info():
        return server_config

    @app.get("/ping", tags=["general"], response_model=bool)
    @app.get("/health", tags=["general"], response_model=bool)
    @app.get("/healthcheck", tags=["general"], response_model=bool)
    @app.get("/status", tags=["general"], response_model=bool)
    def _health():
        return True

    @app.post("/endpoints", tags=["endpoints"], response_model=bool)
    def _add_endpoint_endpoint(cfg: EndpointConfig):
        if cfg.name is None:
            cfg.name = f"endpoint-{len(app.routes)}"
        _add_endpoint(
            app,
            server_config,
            cfg,
            executor,
            context,
            server_logger,
        )
        # force regeneration of the docs
        app.openapi_schema = None
        return True

    @app.delete("/endpoints", tags=["endpoints"], response_model=bool)
    def _delete_endpoint(cfg: EndpointConfig):
        _LOGGER.info(f"Deleting endpoint for {cfg}")
        matching = [r for r in app.routes if r.path == cfg.route]
        assert len(matching) == 1
        app.routes.remove(matching[0])
        # force regeneration of the docs
        app.openapi_schema = None
        return True

    # create pipelines & endpoints
    for endpoint_config in server_config.endpoints:
        _add_endpoint(
            app,
            server_config,
            endpoint_config,
            executor,
            context,
            server_logger,
        )

    _LOGGER.info(f"Added endpoints: {[route.path for route in app.routes]}")

    return app


def _set_pytorch_num_threads(server_config: ServerConfig):
    if server_config.pytorch_num_threads is not None:
        try:
            import torch

            torch.set_num_threads(server_config.pytorch_num_threads)
            _LOGGER.info(f"torch.set_num_threads({server_config.pytorch_num_threads})")
        except ImportError:
            _LOGGER.debug(
                "pytorch not installed, skipping pytorch_num_threads configuration"
            )


def _set_thread_pinning(server_config: ServerConfig):
    pinning = {"core": ("1", "0"), "numa": ("0", "1"), "none": ("0", "0")}

    if server_config.engine_thread_pinning not in pinning:
        raise ValueError(
            "Invalid value for engine_thread_pinning. "
            'Expected one of {"core","numa","none"}. Found '
            f"{server_config.engine_thread_pinning}"
        )

    cores, socks = pinning[server_config.engine_thread_pinning]
    os.environ["NM_BIND_THREADS_TO_CORES"] = cores
    os.environ["NM_BIND_THREADS_TO_SOCKETS"] = socks

    _LOGGER.info(f"NM_BIND_THREADS_TO_CORES={cores}")
    _LOGGER.info(f"NM_BIND_THREADS_TO_SOCKETS={socks}")


def _add_endpoint(
    app: FastAPI,
    server_config: ServerConfig,
    endpoint_config: EndpointConfig,
    executor: ThreadPoolExecutor,
    context: Context,
    server_logger: BaseLogger,
):
    pipeline_config = endpoint_config.to_pipeline_config()
    pipeline_config.kwargs["executor"] = executor

    _LOGGER.info(f"Initializing pipeline for '{endpoint_config.name}'")
    if server_config.integration == INTEGRATION_OPENAI:
        from deepsparse.server.openai_server import DeepSparseOpenAIEngine
        pipeline = DeepSparseOpenAIEngine(pipeline_config=pipeline_config, context=context)
    else:
        pipeline = Pipeline.from_config(pipeline_config, context, server_logger)

    _LOGGER.info(f"Adding endpoints for '{endpoint_config.name}'")
    _add_pipeline_endpoint(
        app,
        endpoint_config,
        server_config.system_logging,
        pipeline,
        server_config.integration,
    )


def _add_pipeline_endpoint(
    app: FastAPI,
    endpoint_config: EndpointConfig,
    system_logging_config: SystemLoggingConfig,
    pipeline: Pipeline,
    integration: str = INTEGRATION_LOCAL,
):
    routes_and_fns = []

    if integration == INTEGRATION_OPENAI:
        from deepsparse.server.openai_server import CompletionResponse, CompletionRequest
        async def create_completion(raw_request: Request):
            """Completion API similar to OpenAI's API.

            See https://platform.openai.com/docs/api-reference/completions/create
            for the API specification. This API mimics the OpenAI Completion API.
            
            """
            request = CompletionRequest(**await raw_request.json())
            _LOGGER.info(f"Received completion request: {request}")

            if request.echo:
                # We do not support echo since we do not
                # currently support getting the logprobs of prompt tokens.
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, "echo is not currently supported"
                )

            if request.suffix is not None:
                # The language models we currently support do not support suffix.
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, "suffix is not currently supported"
                )

            if request.logit_bias is not None:
                # TODO: support logit_bias
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported"
                )

            model_name = pipeline.model
            request_id = f"cmpl-{random_uuid()}"
            if isinstance(request.prompt, list):
                if len(request.prompt) == 0:
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST, "please provide at least one prompt"
                    )
                if len(request.prompt) > 1:
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST,
                        "multiple prompts in a batch is not currently supported",
                    )
                prompt = request.prompt[0]
            else:
                prompt = request.prompt
            created_time = int(time.time())
            try:
                sampling_params = dict(
                    n=request.n,
                    best_of=request.best_of,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop=request.stop,
                    ignore_eos=request.ignore_eos,
                    max_tokens=request.max_tokens,
                    logprobs=request.logprobs,
                    use_beam_search=request.use_beam_search,
                    stream=request.stream,
                )
            except ValueError as e:
                return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

            result_generator = pipeline.generate(prompt, request_id, **sampling_params)

            # Similar to the OpenAI API, when n != best_of, we do not stream the
            # results. In addition, we do not stream the results when use beam search.
            stream = (
                request.stream
                and (request.best_of is None or request.n == request.best_of)
                and not request.use_beam_search
            )

            async def abort_request() -> None:
                await pipeline.abort(request_id)

            def create_stream_response_json(
                index: int,
                text: str,
                logprobs: Optional[LogProbs] = None,
                finish_reason: Optional[str] = None,
            ) -> str:
                choice_data = CompletionResponseStreamChoice(
                    index=index,
                    text=text,
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                )
                response = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[choice_data],
                )
                response_json = response.json(ensure_ascii=False)

                return response_json

            async def completion_stream_generator() -> AsyncGenerator[str, None]:
                previous_texts = [""] * request.n
                previous_num_tokens = [0] * request.n
                async for res in result_generator:
                    res: RequestOutput
                    for output in res.outputs:
                        i = output.index
                        delta_text = output.text[len(previous_texts[i]) :]
                        if request.logprobs is not None:
                            logprobs = create_logprobs(
                                output.token_ids[previous_num_tokens[i] :],
                                output.logprobs[previous_num_tokens[i] :],
                                len(previous_texts[i]),
                            )
                        else:
                            logprobs = None
                        previous_texts[i] = output.text
                        previous_num_tokens[i] = len(output.token_ids)
                        response_json = create_stream_response_json(
                            index=i,
                            text=delta_text,
                            logprobs=logprobs,
                        )
                        yield f"data: {response_json}\n\n"
                        if output.finish_reason is not None:
                            logprobs = LogProbs() if request.logprobs is not None else None
                            response_json = create_stream_response_json(
                                index=i,
                                text="",
                                logprobs=logprobs,
                                finish_reason=output.finish_reason,
                            )
                            yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            # Streaming response
            if stream:
                background_tasks = BackgroundTasks()
                # Abort the request if the client disconnects.
                background_tasks.add_task(abort_request)
                return StreamingResponse(
                    completion_stream_generator(),
                    media_type="text/event-stream",
                    background=background_tasks,
                )

            # Non-streaming response
            final_res: RequestOutput = None
            async for res in result_generator:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await abort_request()
                    return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
                final_res = res
            assert final_res is not None
            choices = []
            for output in final_res.outputs:
                if request.logprobs is not None:
                    logprobs = create_logprobs(output.token_ids, output.logprobs)
                else:
                    logprobs = None
                choice_data = CompletionResponseChoice(
                    index=output.index,
                    text=output.text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                )
                choices.append(choice_data)

            num_prompt_tokens = len(final_res.prompt_token_ids)
            num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            response = CompletionResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=choices,
                usage=usage,
            )

            if request.stream:
                # When user requests streaming but we don't stream, we still need to
                # return a streaming response with a single event.
                response_json = response.json(ensure_ascii=False)

                async def fake_stream_generator() -> AsyncGenerator[str, None]:
                    yield f"data: {response_json}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    fake_stream_generator(), media_type="text/event-stream"
                )

            return response

        output_schema = CompletionResponse

        route = "/completions" + endpoint_config.route
        routes_and_fns.append((route, create_completion))
        
    else:
        input_schema = pipeline.input_schema
        output_schema = pipeline.output_schema

        def _predict(request: pipeline.input_schema):
            pipeline_outputs = pipeline(request)
            server_logger = pipeline.logger
            if server_logger:
                log_system_information(
                    server_logger=server_logger,
                    system_logging_config=system_logging_config,
                )
            pipeline_outputs = prep_outputs_for_serialization(pipeline_outputs)
            return pipeline_outputs

        def _predict_from_files(request: List[UploadFile]):
            request = pipeline.input_schema.from_files(
                (file.file for file in request), from_server=True
            )
            return _predict(request)

    
        if integration == INTEGRATION_LOCAL:
            route = endpoint_config.route or "/predict"
            if not route.startswith("/"):
                route = "/" + route

            routes_and_fns.append((route, _predict))
            if hasattr(input_schema, "from_files"):
                routes_and_fns.append((route + "/from_files", _predict_from_files))
        elif integration == INTEGRATION_SAGEMAKER:
            route = "/invocations"
            if hasattr(input_schema, "from_files"):
                routes_and_fns.append((route, _predict_from_files))
            else:
                routes_and_fns.append((route, _predict))

    for route, endpoint_fn in routes_and_fns:
        app.add_api_route(
            route,
            endpoint_fn,
            response_model=output_schema,
            methods=["POST"],
            tags=["predict"],
        )
        _LOGGER.info(f"Added '{route}' endpoint")
