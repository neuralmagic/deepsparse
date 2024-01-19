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
import numpy
from transformers import AutoTokenizer

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
from scipy.special import softmax


TEST_MODEL_ID = "hf:mgoin/TinyStories-1M-ds"


@pytest.fixture(scope="module")
def endpoint_config():
    endpoint = EndpointConfig(
        task="text_generation",
        model=TEST_MODEL_ID,
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
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


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
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


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
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


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
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


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

    assert (
        response.json()["choices"][0]["text"]
        == 'a was very happy and thanked the man. He said, "Thank you, Sara. You are a '
        + 'good friend."\n\nSara smiled and'
    )


def test_completions_tokenized(client, model_card):
    prompt = "The Boston Bruins are "
    max_tokens = 30

    # Test both passing in input_ids itself as a List[int],
    # and inside of a "batch" as a List[List[int]]
    # TODO: Multiple prompts/batching isn't supported yet
    prefix = "hf:"
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID[len(prefix) :])
    input_ids = tokenizer(prompt).input_ids
    num_prompt_tokens = len(input_ids)

    # Testing both passing in a single prompt tokenized, and it wrapped in a list
    for prompt in [input_ids, [input_ids]]:
        request = CompletionRequest(
            prompt=prompt, max_tokens=max_tokens, model=model_card.id
        )
        response = client.post("/v1/completions", json=request.dict())
        assert response.status_code == 200

        usage = response.json()["usage"]
        assert usage["prompt_tokens"] == num_prompt_tokens
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == max_tokens
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

        assert (
            response.json()["choices"][0]["text"]
            == 'a was very happy and thanked the man. He said, "Thank you, Sara. '
            + 'You are a good friend."\n\nSara smiled and'
        )


def test_logprobs(client, model_card):
    max_tokens = 30
    prompt = "The Boston Bruins are "
    request = CompletionRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        model=model_card.id,
        logprobs=1,
    )
    response = client.post("/v1/completions", json=request.dict())
    assert response.status_code == 200

    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == max_tokens
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    expected_response = (
        'a was very happy and thanked the man. He said, "Thank you, '
        'Sara. You are a good friend."\n\nSara smiled and'
    )
    assert response.json()["choices"][0]["text"] == expected_response

    # Ensure that local pipeline produces the same text and logprobs
    local_model = Pipeline.create(task="text-generation", model=model_card.id)
    output = local_model(prompt=prompt, max_length=max_tokens, output_scores=True)
    assert output.generations[0].text == expected_response

    for local_gen, server_gen in zip(output.generations, response.json()["choices"]):
        local_top1_logprobs = [
            numpy.log(max(softmax(logits))) for logits in local_gen.score
        ]
        server_top1_logprobs = server_gen["logprobs"]["token_logprobs"]
        assert numpy.allclose(local_top1_logprobs, server_top1_logprobs)
