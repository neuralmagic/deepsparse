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


import logging
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional

from deepsparse import Pipeline
from deepsparse.server.config import EndpointConfig
from deepsparse.server.helpers import create_error_response
from deepsparse.server.output import CompletionOutput, RequestOutput
from deepsparse.server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
    random_uuid,
)
from deepsparse.server.server import Server
from deepsparse.tasks import SupportedTasks
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse


_LOGGER = logging.getLogger(__name__)

OPENAI_CHAT_NOT_SUPPORTED = ["logit_bias", "best_ok", "ignore_eos", "use_beam_search"]
OPENAI_TO_DEEPSPARSE_MAPPINGS = {
    "max_tokens": "max_length",
    "frequency_penalty": "repetition_penalty",
}


class OpenAIServer(Server):
    def __init__(self, **kwargs):
        self.model_list = ModelList()
        self.model_to_pipeline = {}

        super().__init__(**kwargs)

    def _add_routes(self, app: FastAPI):
        for endpoint_config in self.server_config.endpoints:
            self._add_model(
                app,
                endpoint_config,
            )

        app.model_list = self.model_list
        app.model_to_pipeline = self.model_to_pipeline

        @app.get("/v1/models", tags=["model"])
        async def show_available_models():
            """Show available models."""
            return app.model_list

        @app.post(
            "/v1/chat/completions",
            tags=["model", "inference"],
            response_model=ChatCompletionResponse,
        )
        async def create_chat_completion(raw_request: Request):
            """Completion API similar to OpenAI's API.

            See  https://platform.openai.com/docs/api-reference/chat/create
            for the API specification. This API mimics the OpenAI ChatCompletion API.
            """
            request = ChatCompletionRequest(**await raw_request.json())
            _LOGGER.info(f"Received chat completion request: {request}")

            prompt = request.messages
            request_id = f"cmpl-{random_uuid()}"
            created_time = int(time.time())
            model = request.model

            pipeline = app.model_to_pipeline.get(model)
            if not pipeline:
                create_error_response(
                    HTTPStatus.BAD_REQUEST, f"{model} is not available"
                )

            try:
                sampling_params = dict(
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                    max_tokens=request.max_tokens,
                    top_k=request.top_k,
                    stream=request.stream,
                    num_return_sequences=request.n,
                )
            except ValueError as e:
                return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

            result_generator = OpenAIServer.generate(
                prompt, request_id, sampling_params, pipeline
            )

            async def abort_request() -> None:
                await pipeline.abort(request_id)

            # Streaming response
            if request.stream:
                background_tasks = BackgroundTasks()
                # Abort the request if the client disconnects.
                background_tasks.add_task(abort_request)
                return StreamingResponse(
                    completion_stream_generator(
                        request,
                        result_generator,
                        request_id=request_id,
                        created_time=created_time,
                        pipeline=pipeline,
                    ),
                    media_type="text/event-stream",
                    background=background_tasks,
                )

            # Non-streaming response
            final_res: RequestOutput = None
            async for res in result_generator:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await abort_request()
                    return OpenAIServer.create_error_response(
                        HTTPStatus.BAD_REQUEST, "Client disconnected"
                    )
                final_res = res
            assert final_res is not None
            choices = []
            for output in final_res.outputs:
                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(role="assistant", content=output.text),
                    finish_reason=output.finish_reason,
                )
                choices.append(choice_data)

            num_prompt_tokens = len(final_res.prompt_token_ids)
            num_generated_tokens = sum(
                len(output.token_ids) for output in final_res.outputs
            )
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            response = ChatCompletionResponse(
                id=request_id,
                created=created_time,
                model=model,
                choices=choices,
                usage=usage,
            )
            return response

        return app

    def _add_model(
        self,
        app: FastAPI,
        endpoint_config: EndpointConfig,
    ):
        pipeline_config = endpoint_config.to_pipeline_config()
        pipeline_config.kwargs["executor"] = self.executor

        _LOGGER.info(f"Initializing pipeline for '{endpoint_config.name}'")

        if not SupportedTasks.is_text_generation(pipeline_config.task):
            raise ValueError(
                "OpenAI API is only available for one of the following "
                f"tasks: {SupportedTasks.text_generation._fields}"
            )

        pipeline = Pipeline.from_config(
            pipeline_config, self.context, self.server_logger
        )

        if not self.model_to_pipeline.get(endpoint_config.model):
            model_card = ModelCard(
                id=endpoint_config.model,
                root=endpoint_config.model,
                permission=[ModelPermission()],
            )

            self.model_to_pipeline[endpoint_config.model] = pipeline
            self.model_list.data.extend(model_card)

    @staticmethod
    async def generate(
        prompt: str, request_id: str, generation_kwargs: dict, pipeline: Pipeline
    ) -> AsyncGenerator[RequestOutput, None]:
        def tokenize(text: str) -> List[int]:
            return pipeline.tokenizer(text)

        prompt_token_ids = tokenize(prompt)
        generation_kwargs = map_generation_schema(generation_kwargs)

        stream = generation_kwargs.pop("stream")
        presence_penalty = generation_kwargs.pop("presence_penalty")
        stop = generation_kwargs.pop("stop")

        output = pipeline(
            sequences=prompt,
            generation_config=generation_kwargs,
            streaming=stream,
            presence_penalty=presence_penalty,
            stop=stop,
        )

        if not stream:
            # Non-streaming responses
            generations = output.generations[0]
            if not isinstance(generations, list):
                generations = [generations]

            generated_outputs = []
            for prompt_generation in generations:
                completion = CompletionOutput(
                    index=0,
                    text=prompt_generation.text,
                    token_ids=tokenize(prompt_generation.text),
                    finish_reason=prompt_generation.finished_reason,
                )
                generated_outputs.append(completion)
            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                outputs=generated_outputs,
                finished=True,
            )
        else:
            concat_text = ""
            concat_token_ids = []
            for generation in output:
                output = generation.generations[0]
                concat_text += output.text
                concat_token_ids.append(tokenize(output.text))
                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=concat_text,
                            token_ids=concat_token_ids,
                            finish_reason=output.finished_reason,
                        )
                    ],
                    finished=output.finished,
                )


def map_generation_schema(generation_kwargs: Dict) -> Dict:
    """
    Map the ChatCompletionRequest to the TextGenerationInput.
    :param generation_kwargs input fields given as part of the ChatCompletionRequest
    :returns: updated generated_kwargs, mapped to the TextGenerationInput while
    raising errors for any properties which are not yet supported.
    """
    for k in list(generation_kwargs.keys()):
        if k in OPENAI_CHAT_NOT_SUPPORTED:
            return create_error_response(
                HTTPStatus.BAD_REQUEST, f"{k} is not currently supported"
            )
        if k in OPENAI_TO_DEEPSPARSE_MAPPINGS:
            generation_kwargs[OPENAI_TO_DEEPSPARSE_MAPPINGS[k]] = generation_kwargs[k]

    if generation_kwargs["num_return_sequences"] > 1:
        generation_kwargs["do_sample"] = True

    return generation_kwargs


def create_stream_response_json(
    index: int,
    text: str,
    request_id: str,
    created_time: int,
    pipeline: Pipeline,
    finish_reason: Optional[str] = None,
) -> str:
    choice_data = ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(content=text),
        finish_reason=finish_reason,
    )
    response = ChatCompletionStreamResponse(
        id=request_id,
        created=created_time,
        model=pipeline.model_path,
        choices=[choice_data],
    )
    response_json = response.json(ensure_ascii=False)

    return response_json


async def completion_stream_generator(
    request, result_generator, request_id, created_time, pipeline
) -> AsyncGenerator[str, None]:
    # First chunk with role
    for i in range(request.n):
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=request_id, choices=[choice_data], model=pipeline.model_path
        )
        data = chunk.json(exclude_unset=True, ensure_ascii=False)
        yield f"data: {data}\n\n"

    previous_texts = [""] * request.n
    previous_num_tokens = [0] * request.n
    async for res in result_generator:
        res: RequestOutput
        for output in res.outputs:
            i = output.index
            delta_text = output.text[len(previous_texts[i]) :]
            previous_texts[i] = output.text
            previous_num_tokens[i] = len(output.token_ids)
            response_json = create_stream_response_json(
                index=i,
                text=delta_text,
                request_id=request_id,
                created_time=created_time,
                pipeline=pipeline,
            )
            yield f"data: {response_json}\n\n"
            if output.finish_reason is not None:
                response_json = create_stream_response_json(
                    index=i,
                    text="",
                    finish_reason=output.finish_reason,
                    request_id=request_id,
                    created_time=created_time,
                    pipeline=pipeline,
                )
                yield f"data: {response_json}\n\n"
    yield "data: [DONE]\n\n"
