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
from http import HTTPStatus
from typing import AsyncGenerator, Dict

from deepsparse.server.output import *
from deepsparse.server.protocol import *
from deepsparse.server.server import Pipeline, Server
from deepsparse.transformers.pipelines.text_generation import TextGenerationInput
from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse


_LOGGER = logging.getLogger(__name__)

OPENAI_CHAT_NOT_SUPPORTED = ["logit_bias", "best_ok", "ignore_eos", "use_beam_search"]
OPENAI_TO_DEEPSPARSE_MAPPINGS = {
    "max_tokens": "max_length",
    "frequency_penalty": "repetition_penalty",
}


class OpenAIServer(Server):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        endpoint_config = self.server_config.endpoints[0]
        pipeline_config = endpoint_config.to_pipeline_config()
        pipeline_config.kwargs["executor"] = self.executor
        self.pipeline = Pipeline.from_config(pipeline_config, self.context)
        self.served_model = self.pipeline.model_path

    def tokenize(self, text: str) -> List[int]:
        return self.pipeline.tokenizer(text)

    def create_error_response(
        self, status_code: HTTPStatus, message: str
    ) -> JSONResponse:
        return JSONResponse(
            ErrorResponse(message=message, type="invalid_request_error").dict(),
            status_code=status_code.value,
        )

    def map_generation_schema(self, generation_kwargs: Dict) -> Dict:
        """
        Map the ChatCompletionRequest to the TextGenerationInput.
        :param generation_kwargs input fields given as part of the ChatCompletionRequest
        :returns: updated generated_kwargs, mapped to the TextGenerationInput while
        raising errors for any properties which are not yet supported.
        """
        for k in list(generation_kwargs.keys()):
            if k in OPENAI_CHAT_NOT_SUPPORTED:
                return self.create_error_response(
                    HTTPStatus.BAD_REQUEST, f"{k} is not currently supported"
                )
            if k in OPENAI_TO_DEEPSPARSE_MAPPINGS:
                generation_kwargs[OPENAI_TO_DEEPSPARSE_MAPPINGS[k]] = generation_kwargs[
                    k
                ]

        return generation_kwargs

    def create_stream_response_json(
        self,
        index: int,
        text: str,
        request_id: str,
        created_time: int,
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
            model=self.served_model,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator(
        self, request, result_generator, request_id, created_time
    ) -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=request_id, choices=[choice_data], model=self.served_model
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
                response_json = self.create_stream_response_json(
                    index=i,
                    text=delta_text,
                    request_id=request_id,
                    created_time=created_time,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    response_json = self.create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                        request_id=request_id,
                        created_time=created_time,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    async def generate(
        self,
        prompt: str,
        request_id: str,
        generation_kwargs: dict,
    ) -> AsyncGenerator[RequestOutput, None]:

        prompt_token_ids = self.tokenize(prompt)
        generation_kwargs = self.map_generation_schema(generation_kwargs)

        stream = generation_kwargs["stream"]
        presence_penalty = generation_kwargs["presence_penalty"]
        stop = generation_kwargs["stop"]

        generation_kwargs.pop("stream")
        generation_kwargs.pop("presence_penalty")
        generation_kwargs.pop("stop")

        output = self.pipeline(
            sequences=prompt,
            generation_config=generation_kwargs,
            streaming=stream,
            presence_penalty=presence_penalty,
            stop=stop,
        )

        if not stream:
            # Non-streaming response
            generation = output.generations[0]
            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=generation.text,
                        token_ids=self.tokenize(generation.text),
                        finish_reason=generation.finished_reason,
                    )
                ],
                finished=generation.finished,
            )
        else:
            concat_text = ""
            concat_token_ids = []
            for generation in output:
                output = generation.generations[0]
                concat_text += output.text
                concat_token_ids.append(self.tokenize(output.text))
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

    def _add_routes(self, app):
        @app.get("/v2/models")
        async def show_available_models():
            """Show available models. Right now we only have one model."""
            model_cards = [
                ModelCard(
                    id=self.served_model,
                    root=self.served_model,
                    permission=[ModelPermission()],
                )
            ]
            return ModelList(data=model_cards)

        @app.post("/v2/chat/completions")
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
                )
            except ValueError as e:
                return self.create_error_response(HTTPStatus.BAD_REQUEST, str(e))

            result_generator = self.generate(prompt, request_id, sampling_params)

            async def abort_request() -> None:
                await engine.abort(request_id)

            # Streaming response
            if request.stream:
                background_tasks = BackgroundTasks()
                # Abort the request if the client disconnects.
                background_tasks.add_task(abort_request)
                return StreamingResponse(
                    self.completion_stream_generator(
                        request,
                        result_generator,
                        request_id=request_id,
                        created_time=created_time,
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
                    return create_error_response(
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
                model=self.served_model,
                choices=choices,
                usage=usage,
            )
            return response

        return app
