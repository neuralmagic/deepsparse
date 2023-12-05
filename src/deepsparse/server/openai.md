## 🔌 OpenAI Integration

The [OpenAI API](https://platform.openai.com/docs/api-reference/introduction) can be used to interact with `text_generation` models.
Similar to the Deepsparse server (see `README.md` for details), a config file can be
created for the text generation models. A sample config is provided below

```yaml
endpoints:
  - task: text_generation
    model: zoo:opt-1.3b-opt_pretrain-pruned50_quantW8A8
```

To start the server with the OpenAI integration, the following command can be used:

`deepsparse.openai sample_config.yaml`

The standard deepsparse server command is also available:

`deepsparse.server --config_file sample_config.yaml --integration openai`

Once launched, the OpenAI endpoints will be available. The payload expected by the endpoints
can be found under the OpenAI documentation for each endpoint. Currently, the supported endpoints
are:

```
/v1/models
/v1/chat/completions
```

Inference requests can be sent through standard curls commands, the requests library,
or through the OpenAI API.

---

### OpenAI API Requests

- Starting the server with the config above, we have access to one model
`zoo:opt-1.3b-opt_pretrain-pruned50_quantW8A8`. We can send an inference request using
the OpenAI API, as shown in the example code below:

```python
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:5543/v1"

stream = False
completion = openai.ChatCompletion.create(
    messages="how are you?",
    stream=stream,
    max_tokens=30,
    model="zoo:opt-1.3b-opt_pretrain-pruned50_quantW8A8",
)

print("Chat results:")
if stream:
    text = ""
    for c in completion:
        print(c)
else:
    print(completion)
```

- We can toggle the `stream` flag to enable streaming outputs as well

---

## Using `curl` or `requests`

- We can also run inference through `curl` commands or by using the `requests` library

`curl`:
```bash
curl http://localhost:5543/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zoo:opt-1.3b-opt_pretrain-pruned50_quantW8A8",
        "messages": "your favourite book?",
        "max_tokens": 30,
        "n": 2,
        "stream": true
    }'
```

`reqeusts`:

```python
import requests

url = "http://localhost:5543/v1/chat/completions"

obj = {
    "model": "zoo:opt-1.3b-opt_pretrain-pruned50_quantW8A8",
    "messages": "how are you?",
    "max_tokens": 10
}

response = requests.post(url, json=obj)
print(response.text)
```
