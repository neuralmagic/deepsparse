# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

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
import json
import logging
import time
import uuid
from http import HTTPStatus
from threading import Thread
from typing import AsyncGenerator, Dict, List, Optional

from packaging import version
from transformers import TextIteratorStreamer

import deepsparse
import fastapi
import uvicorn
from fastapi import FastAPI
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
from deepsparse.engine import Context
from deepsparse.server.config import ServerConfig
from deepsparse.pipeline import Pipeline, PipelineConfig

TIMEOUT_KEEP_ALIVE = 5  # seconds
SUPPORTED_TASKS = ["text_generation", "opt", "codegen", "bloom"]

_LOGGER = logging.getLogger(__name__)

__all__ = ["DeepSparseOpenAIEngine", "cre"]

class DeepSparseOpenAIEngine:
    def __init__(
        self,
        context: Context,
        pipeline_config: PipelineConfig
    ):
        self.engine = Pipeline.from_config(pipeline_config, context)
        self.model = self.engine.model_path
        self.tokenizer = self.engine.tokenizer
        
    def tokenize(self, text: str) -> List[int]:
        return self.engine.tokenizer(text)

    def generate(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int = 64,
        top_p: float = 0.95,
        temperature: float = 0.80,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = True,
        **kwargs,
    ) -> AsyncGenerator[RequestOutput, None]:
        request_id = random_uuid()

        prompt_token_ids = self.tokenize(prompt)

        self.engine.max_generated_tokens = max_tokens
        self.engine.sampling_temperature = temperature

        stream = False
        if not stream:
            # Non-streaming response
            output = self.engine(sequences=prompt)
            new_text = output.sequences[0]

            print("RETURNING")
            return RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=new_text,
                        token_ids=self.tokenize(new_text),
                        finish_reason="stop",
                    )
                ],
                finished=True,
            )

        else:
            # Streaming response
            streamer = TextIteratorStreamer(self.engine.tokenizer)

            generation_kwargs = dict(sequences=prompt, streamer=streamer)

            thread = Thread(target=self.engine, kwargs=generation_kwargs)
            thread.start()

            # stream out the text
            concat_text = ""
            concat_token_ids = []
            for new_text in streamer:
                concat_text += new_text
                concat_token_ids.append(self.tokenize(new_text))
                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    outputs=[
                        CompletionOutput(
                            index=0, text=concat_text, token_ids=concat_token_ids
                        )
                    ],
                    finished=False,
                )

            # finished
            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                outputs=[
                    CompletionOutput(
                        index=0, text="", token_ids=[0], finish_reason="stop"
                    )
                ],
                finished=True,
            )

    def abort(self, session_id):
        pass


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )

def create_logprobs(
    token_ids: List[int],
    id_logprobs: List[Dict[int, float]],
    initial_text_offset: int = 0,
    pipeline: DeepSparseOpenAIEngine = None,
) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = pipeline.tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(id_logprob[token_id])
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
        last_token_len = len(token)

        logprobs.top_logprobs.append(
            {pipeline.tokenizer.convert_ids_to_tokens(i): p for i, p in id_logprob.items()}
        )
    return logprobs

def create_completion(request: CompletionRequest, pipeline: DeepSparseOpenAIEngine):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.
    """
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

    def abort_request() -> None:
        pipeline.abort(request_id)

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

    def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]) :]
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        output.token_ids[previous_num_tokens[i] :],
                        output.logprobs[previous_num_tokens[i] :],
                        len(previous_texts[i]),
                        pipeline=pipeline
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
    #for res in result_generator:
        #if raw_request.is_disconnected():
            # Abort the request if the client disconnects.
        #    abort_request()
        #    return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
        #final_res = res
    final_res = result_generator
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        if request.logprobs is not None:
            logprobs = create_logprobs(output.token_ids, output.logprobs, pipeline=pipeline)
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

        def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            fake_stream_generator(), media_type="text/event-stream"
        )

    return response