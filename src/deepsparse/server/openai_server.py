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

import numpy

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
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
    random_uuid,
)
from deepsparse.server.server import Server
from deepsparse.tasks import SupportedTasks
from deepsparse.utils import InferenceState, numpy_softmax
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse


_LOGGER = logging.getLogger(__name__)

OPENAI_CHAT_NOT_SUPPORTED = ["logit_bias", "best_ok", "ignore_eos", "use_beam_search"]
OPENAI_TO_DEEPSPARSE_MAPPINGS = {
    "max_tokens": "max_length",
    "frequency_penalty": "repetition_penalty",
}


def apply_chatml_chat_template(messages: List[Dict[str, str]]) -> str:
    # When there is no chat template available, use ChatML as the default
    # https://github.com/openai/openai-python/blob/release-v0.28.1/chatml.md
    prompt = ""
    for message in messages:
        prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return prompt


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
            # Completion API similar to OpenAI's API.

            # See  https://platform.openai.com/docs/api-reference/chat/create
            # for the API specification. This API mimics the OpenAI ChatCompletion API.

            request = ChatCompletionRequest(**await raw_request.json())
            _LOGGER.debug("Received chat completion request %s" % request)

            model = request.model
            pipeline = app.model_to_pipeline.get(model)
            if not pipeline:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    f"The model `{model}` does not exist.",
                )

            messages = request.messages
            # For chat templating, the message needs to be formatted
            # as a list of dictionaries of `{"role": "", "content": ""}`
            # https://huggingface.co/docs/transformers/chat_templating
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            try:
                if hasattr(pipeline.tokenizer, "apply_chat_template"):
                    prompt = pipeline.tokenizer.apply_chat_template(
                        conversation=messages,
                        add_generation_prompt=request.add_generation_prompt,
                        tokenize=False,
                    )
                else:
                    # tokenizer.apply_chat_template requires Transformers>=4.34, so
                    # if it is not available, default to standard chatml
                    _LOGGER.warning(
                        "Cannot use tokenizer.apply_chat_template, please update to "
                        "transformers>=4.34 for best chat results. Defaulting to ChatML"
                    )
                    prompt = apply_chatml_chat_template(messages=messages)
            except Exception as e:
                _LOGGER.error(f"Error in applying chat template from request: {str(e)}")
                return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

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
                prompt, request_id, sampling_params, pipeline
            )

            # Streaming response
            if request.stream:
                background_tasks = BackgroundTasks()
                # Abort the request if the client disconnects.
                return StreamingResponse(
                    chat_completion_stream_generator(
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
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST, "Client disconnected"
                    )
                final_res = res
            assert final_res is not None
            choices = []
            for output in final_res.outputs:
                choice_data = ChatCompletionResponseChoice(
                    message=ChatMessage(role="assistant", content=output.text),
                    finish_reason=output.finish_reason,
                )
                choices.append(choice_data)

            # Note: indexing 0 for now as handling singular prompt
            num_prompt_tokens = len(final_res.prompt_token_ids["input_ids"][0])
            num_generated_tokens = sum(
                len(output.token_ids["input_ids"]) for output in final_res.outputs
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

        @app.post(
            "/v1/completions",
            tags=["model", "inference"],
            response_model=CompletionResponse,
        )
        async def create_completion(raw_request: Request):
            request = CompletionRequest(**await raw_request.json())
            _LOGGER.debug("Received completion request: %s" % request)

            model = request.model

            pipeline = app.model_to_pipeline.get(model)
            if not pipeline:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    f"The model `{model}` does not exist.",
                )

            request_id = f"cmpl-{random_uuid()}"
            created_time = int(time.time())

            if isinstance(request.prompt, list):
                if len(request.prompt) == 0:
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST, "please provide at least one prompt"
                    )
                first_element = request.prompt[0]
                if isinstance(first_element, int):
                    # There is just a single tokenized prompt
                    prompt = pipeline.tokenizer.decode(request.prompt)
                elif isinstance(first_element, (str, list)):
                    if len(request.prompt) > 1:
                        return create_error_response(
                            HTTPStatus.BAD_REQUEST,
                            "multiple prompts in a batch is not currently supported",
                        )
                    if isinstance(first_element[0], int):
                        prompt = pipeline.tokenizer.decode(first_element)
                    else:
                        prompt = first_element
            else:
                prompt = request.prompt

            try:
                sampling_params = dict(
                    num_return_sequences=request.n,
                    output_scores=request.logprobs,
                    presence_penalty=request.presence_penalty,
                    frequency_penalty=request.frequency_penalty,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop=request.stop,
                    max_tokens=request.max_tokens,
                    stream=request.stream,
                )
            except ValueError as e:
                return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

            result_generator = OpenAIServer.generate(
                prompt=prompt,
                request_id=request_id,
                generation_kwargs=sampling_params,
                pipeline=pipeline,
            )

            # Streaming response
            if request.stream:
                background_tasks = BackgroundTasks()
                # Abort the request if the client disconnects.
                return StreamingResponse(
                    completion_stream_generator(
                        request=request,
                        request_id=request_id,
                        result_generator=result_generator,
                        pipeline=pipeline,
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
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST, "Client disconnected"
                    )
                final_res = res
            assert final_res is not None
            choices = []
            for output in final_res.outputs:
                logprobs = (
                    create_logprobs(
                        output.token_ids["input_ids"],
                        output.scores,
                        pipeline=pipeline,
                    )
                    if request.logprobs
                    else None
                )
                choice_data = CompletionResponseChoice(
                    text=output.text,
                    finish_reason=output.finish_reason,
                    logprobs=logprobs,
                )
                choices.append(choice_data)

            # Note: indexing 0 for now as handling singular prompt
            num_prompt_tokens = len(final_res.prompt_token_ids["input_ids"][0])
            num_generated_tokens = sum(
                len(output.token_ids["input_ids"]) for output in final_res.outputs
            )
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            response = CompletionResponse(
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
        """
        Function to add models to the app. All models are added when the server is
        first launched and stored as individual ModelCard objects in the model_list
        attribute. Mapping between the model identifier (i.e the model_path/zoostub)
        and pipeline is stored in the model_to_pipeline attribute.

        :param app: FastAPI app
        :param endpoint_config: endpoint config for the specific model being added
        """
        pipeline_config = endpoint_config.to_pipeline_config()

        _LOGGER.debug("Initializing pipeline for %s" % endpoint_config.name)

        if not (
            SupportedTasks.is_text_generation(pipeline_config.task)
            or SupportedTasks.is_code_generation(pipeline_config.task)
        ):
            raise ValueError(
                "OpenAI API is only available for one of the following "
                f"tasks: {SupportedTasks.text_generation._fields}, "
                f"{SupportedTasks.code_generation._fields}"
            )

        if pipeline_config.kwargs.get("continuous_batch_sizes"):
            _LOGGER.info(
                "for continuous batching, the single stream scheduler will be enabled."
            )
            pipeline_config.num_cores = self.server_config.num_cores
            pipeline_config.scheduler = "single"

            pipeline = Pipeline.from_config(
                pipeline_config,
                num_streams=self.server_config.num_workers,
            )
        else:
            pipeline = Pipeline.from_config(pipeline_config, context=self.context)

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
        """
        Map the request input to the TextGenerationInput schema using the
        map_generation_schema function. Call the pipeline and return the generations
        using a generator

        :param prompt: prompt to run inference on
        :param request_id: request_id for the specific inference call
        :param generation_kwargs: generation_kwargs for inference
        :para pipeline: TextGenerationPipeline object

        :return: generator consisting of each of the generations
        """

        def tokenize(text: str, return_tensors=None) -> Dict[str, List[int]]:
            if return_tensors:
                return pipeline.tokenizer(text, return_tensors=return_tensors)
            return pipeline.tokenizer(text)

        generation_kwargs = map_generation_schema(generation_kwargs)

        stream = generation_kwargs.pop("stream")
        presence_penalty = generation_kwargs.pop("presence_penalty")
        stop = generation_kwargs.pop("stop")

        inference_state = InferenceState()
        inference_state.create_state({})

        output = await pipeline.run_async(
            inference_state=inference_state,
            sequences=prompt,
            generation_kwargs=generation_kwargs,
            streaming=stream,
            return_input_tokens=True,
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
                    scores=prompt_generation.score,
                    text=prompt_generation.text,
                    token_ids=tokenize(prompt_generation.text),
                    finish_reason=prompt_generation.finished_reason,
                )
                generated_outputs.append(completion)

            yield RequestOutput(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=output.input_tokens,
                outputs=generated_outputs,
                finished=True,
            )
        else:
            concat_token_ids = []
            async for generation in output:
                output = generation.generations[0]
                concat_token_ids.append(tokenize(output.text, return_tensors="np"))
                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=generation.input_tokens,
                    outputs=[
                        CompletionOutput(
                            scores=output.score,
                            text=output.text,
                            token_ids=concat_token_ids,
                            finish_reason=output.finished_reason,
                        )
                    ],
                    finished=output.finished,
                )


def create_logprobs(
    token_ids: List[int], scores: numpy.ndarray, pipeline: Pipeline
) -> LogProbs:

    logprobs = LogProbs()
    tokens = pipeline.tokenizer.batch_decode(token_ids)

    for i in range(len(tokens)):
        log_prob = float(numpy.log(max(numpy_softmax(scores[i]))))
        logprobs.tokens.append(tokens[i])
        logprobs.token_logprobs.append(log_prob)

    return logprobs


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
            generation_kwargs.pop(k)

    if generation_kwargs["num_return_sequences"] > 1:
        generation_kwargs["do_sample"] = True

    return generation_kwargs


def create_stream_response_json(
    text: str,
    request_id: str,
    created_time: int,
    pipeline: Pipeline,
    finish_reason: Optional[str] = None,
) -> str:
    """
    Create the response for /v1/chat/completions endpoint when streaming is enabled.
    """
    choice_data = ChatCompletionResponseStreamChoice(
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


def create_completion_stream_response_json(
    text: str,
    request_id: str,
    created_time: int,
    pipeline: Pipeline,
    finish_reason: Optional[str] = None,
    logprobs: Optional[LogProbs] = None,
) -> str:
    """
    Create the response for /v1/completions endpoint when streaming is enabled.
    """
    choice_data = CompletionResponseStreamChoice(
        text=text,
        logprobs=logprobs,
        finish_reason=finish_reason,
    )
    response = CompletionStreamResponse(
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
    async for res in result_generator:
        res: RequestOutput
        for output in res.outputs:
            logprobs = (
                create_logprobs(
                    output.token_ids[-1]["input_ids"],
                    output.scores[0],
                    pipeline=pipeline,
                )
                if request.logprobs
                else None
            )
            response_json = create_completion_stream_response_json(
                text=output.text,
                request_id=request_id,
                created_time=created_time,
                pipeline=pipeline,
                logprobs=logprobs,
                finish_reason=output.finish_reason,
            )
            yield f"data: {response_json}\n\n"
    yield "data: [DONE]\n\n"


async def chat_completion_stream_generator(
    request, result_generator, request_id, created_time, pipeline
) -> AsyncGenerator[str, None]:
    # First chunk with role
    for i in range(request.n):
        choice_data = ChatCompletionResponseStreamChoice(
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=request_id, choices=[choice_data], model=pipeline.model_path
        )
        data = chunk.json(exclude_unset=True, ensure_ascii=False)
        yield f"data: {data}\n\n"

    async for res in result_generator:
        res: RequestOutput
        for output in res.outputs:
            response_json = create_stream_response_json(
                text=output.text,
                request_id=request_id,
                created_time=created_time,
                pipeline=pipeline,
                finish_reason=output.finish_reason,
            )
            yield f"data: {response_json}\n\n"
    yield "data: [DONE]\n\n"
