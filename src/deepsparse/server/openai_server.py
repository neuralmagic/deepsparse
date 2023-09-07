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

__all__ = ["DeepSparseOpenAIEngine"]

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

    async def generate(
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

        if not stream:
            # Non-streaming response
            output = self.engine(sequences=prompt)
            new_text = output.sequences[0]

            yield RequestOutput(
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

    async def abort(self, session_id):
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