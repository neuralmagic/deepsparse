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
from functools import partial
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
from deepsparse.server.server import ProxyPipeline, Server
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse


_LOGGER = logging.getLogger(__name__)

OPENAI_CHAT_NOT_SUPPORTED = ["logit_bias", "best_ok", "ignore_eos", "use_beam_search"]
OPENAI_TO_DEEPSPARSE_MAPPINGS = {
    "max_tokens": "max_length",
    "frequency_penalty": "repetition_penalty",
}
SUPPORTED_TASKS = ["text_generation", "opt", "bloom"]


class OpenAIServer(Server):
    def _add_routes(self, app: FastAPI):
        for endpoint_config in self.server_config.endpoints:
            self._add_endpoint(
                app,
                endpoint_config,
            )

        _LOGGER.info(f"Added endpoints: {[route.path for route in app.routes]}")
        return app

    def _add_endpoint(
        self,
        app: FastAPI,
        endpoint_config: EndpointConfig,
    ):
        pipeline_config = endpoint_config.to_pipeline_config()
        pipeline_config.kwargs["executor"] = self.executor

        _LOGGER.info(f"Initializing pipeline for '{endpoint_config.name}'")

        if pipeline_config.task not in SUPPORTED_TASKS:
            raise ValueError(
                "OpenAI API is only available for one of the following "
                f"tasks: {SUPPORTED_TASKS}"
            )

        pipeline = Pipeline.from_config(
            pipeline_config, self.context, self.server_logger
        )

        _LOGGER.info(f"Adding endpoints for '{endpoint_config.name}'")
        self._add_chat_completion_endpoint(
            app,
            endpoint_config,
            pipeline,
        )

    def _add_chat_completion_endpoint(
        self,
        app: FastAPI,
        endpoint_config: EndpointConfig,
        pipeline: Pipeline,
    ):
        routes_and_fns = []
        route = (
            f"{endpoint_config.route}/chat/completions"
            if endpoint_config.route
            else f"/v2/models/{endpoint_config.name}/chat/completions"
        )
        route = self.clean_up_route(route)
        routes_and_fns.append(
            (
                route,
                partial(OpenAIServer.create_chat_completion, ProxyPipeline(pipeline)),
            )
        )

        self._update_routes(
            app=app,
            routes_and_fns=routes_and_fns,
            response_model=ChatCompletionResponse,
            methods=["POST"],
            tags=["model", "inference"],
        )

    @staticmethod
    async def generate(
        prompt: str, request_id: str, generation_kwargs: dict, pipeline: Pipeline
    ) -> AsyncGenerator[RequestOutput, None]:
        def tokenize(text: str) -> List[int]:
            return pipeline.tokenizer(text)

        prompt_token_ids = tokenize(prompt)
        generation_kwargs = map_generation_schema(generation_kwargs)

        stream = generation_kwargs["stream"]
        presence_penalty = generation_kwargs["presence_penalty"]
        stop = generation_kwargs["stop"]

        generation_kwargs.pop("stream")
        generation_kwargs.pop("presence_penalty")
        generation_kwargs.pop("stop")

        output = pipeline(
            sequences=prompt,
            generation_config=generation_kwargs,
            streaming=stream,
            presence_penalty=presence_penalty,
            stop=stop,
        )

        if not stream:
            # Non-streaming responss
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

    @staticmethod
    async def show_available_models(proxy_pipeline: ProxyPipeline):
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(
                id=proxy_pipeline.pipeline.model_path,
                root=proxy_pipeline.pipeline.model_path,
                permission=[ModelPermission()],
            )
        ]
        return ModelList(data=model_cards)

    @staticmethod
    async def create_chat_completion(
        proxy_pipeline: ProxyPipeline, raw_request: Request
    ):
        """Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.
        """
        request = ChatCompletionRequest(**await raw_request.json())
        _LOGGER.info(f"Received chat completion request: {request}")

        prompt = request.messages
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())

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
            prompt, request_id, sampling_params, proxy_pipeline.pipeline
        )

        async def abort_request() -> None:
            await proxy_pipeline.pipeline.abort(request_id)

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
                    pipeline=proxy_pipeline.pipeline,
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
            model=proxy_pipeline.pipeline.model_path,
            choices=choices,
            usage=usage,
        )
        return response


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
