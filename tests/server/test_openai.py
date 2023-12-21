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
import pytest
from deepsparse import Pipeline
from deepsparse.server.config import EndpointConfig, ServerConfig
from deepsparse.server.openai_server import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelPermission,
    OpenAIServer,
)
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def endpoint_config():
    endpoint = EndpointConfig(
        task="text_generation",
        model="hf:mgoin/TinyStories-1M-ds",
    )
    return endpoint


@pytest.fixture(scope="module")
def model_card(endpoint_config):
    return ModelCard(
        id=endpoint_config.model,
        root=endpoint_config.model,
        permission=[ModelPermission()],
    )


@pytest.fixture(scope="module")
def server_config(endpoint_config):
    server_config = ServerConfig(
        num_cores=1, num_workers=1, endpoints=[endpoint_config], loggers={}
    )
    return server_config


@pytest.fixture(scope="module")
def server(server_config):
    server = OpenAIServer(server_config=server_config)
    return server


@pytest.fixture(scope="module")
def app(server):
    app = server._build_app()
    return app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


def test_openai_server_creation(app):
    assert app.routes[-1].path == "/v1/completions"
    assert app.routes[-2].path == "/v1/chat/completions"
    assert app.routes[-3].path == "/v1/models"


def test_correct_models_added_to_model_list(app, server, model_card):
    assert server.model_list.data[0][-1] == model_card.id


def test_add_same_model(app, server, endpoint_config):
    server._add_model(app, endpoint_config)
    assert len(list(server.model_to_pipeline.keys())) == 1
    assert isinstance(server.model_to_pipeline[endpoint_config.model], Pipeline)


def test_get_models(client, model_card):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json().get("data")[0][-1] == model_card.id


def test_chat_completions_string(client, model_card):
    max_tokens = 15
    request = ChatCompletionRequest(
        messages="How is the weather in Boston?",
        max_tokens=max_tokens,
        model=model_card.id,
    )
    response = client.post("/v1/chat/completions", json=request.dict())
    assert response.status_code == 200

    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    message = response.json()["choices"][0]["message"]
    assert message["content"] == "\n\n\nPossible story:\n\nLily and Ben were playing in"


def test_chat_completions_dict(client, model_card):
    max_tokens = 15
    request = ChatCompletionRequest(
        messages={"role": "user", "content": "How is the weather in Boston?"},
        max_tokens=max_tokens,
        model=model_card.id,
    )
    response = client.post("/v1/chat/completions", json=request.dict())
    assert response.status_code == 200

    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    message = response.json()["choices"][0]["message"]
    assert message["content"] == "\n\n\nPossible story:\n\nLily and Ben were playing in"


def test_chat_completions_list(client, model_card):
    max_tokens = 15
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "How is the weather in Boston?"}],
        max_tokens=max_tokens,
        model=model_card.id,
    )
    response = client.post("/v1/chat/completions", json=request.dict())
    assert response.status_code == 200

    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 8
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    message = response.json()["choices"][0]["message"]
    assert message["content"] == "\n\n\nPossible story:\n\nLily and Ben were playing in"


def test_chat_completions_multiturn(client, model_card):
    max_tokens = 20
    request = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi back!"},
            {"role": "user", "content": "I like talking with you."},
        ],
        max_tokens=max_tokens,
        model=model_card.id,
    )
    response = client.post("/v1/chat/completions", json=request.dict())
    assert response.status_code == 200

    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 21
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completions_fastchat_list(client, model_card):
    request = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        max_tokens=50,
        model=model_card.id,
    )
    response = client.post("/v1/chat/completions", json=request.dict())
    assert response.status_code == 200


def test_completions(client, model_card):
    max_tokens = 30
    request = CompletionRequest(
        prompt="The Boston Bruins are ", max_tokens=max_tokens, model=model_card.id
    )
    response = client.post("/v1/completions", json=request.dict())
    assert response.status_code == 200

    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
