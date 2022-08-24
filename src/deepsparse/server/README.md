## 🔌 DeepSparse Server

```bash
pip install deepsparse[server]
```

The DeepSparse server allows you to serve models and pipelines for deployment in HTTP. The server runs on top of the popular FastAPI web framework and Uvicorn web server.
The server supports any task from deepsparse. Pipeline including NLP, image classification, and object detection tasks.
An updated list of available tasks can be found
[here](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/PIPELINES.md)

 - Run `deepsparse.server --help` to lookup the available CLI arguments.

```
Usage: deepsparse.server [OPTIONS] COMMAND [ARGS]...

  Start a DeepSparse inference server for serving the models and pipelines.

      1. `deepsparse.server config [OPTIONS] <config path>`

      2. `deepsparse.server task [OPTIONS] <task>

  Examples for using the server:

      `deepsparse.server config server-config.yaml`

      `deepsparse.server task question_answering --batch-size 2`

      `deepsparse.server task question_answering --host "0.0.0.0"`

  Example config.yaml for serving:

  ```yaml
  num_cores: 2
  num_workers: 2
  endpoints:
    - task: question_answering
      endpoint: /unpruned/predict
      model_path: zoo:some/zoo/stub
    - task: question_answering
      endpoint: /pruned/predict
      model_path: /path/to/local/model
  ```

Options:
  --help  Show this message and exit.

Commands:
  config  Run the server using configuration from a .yaml file.
  task    Run the server using configuration with CLI options, which can...
```

### Single Model Inference

Example CLI command for serving a single model for the **question answering** task:

```bash
deepsparse.server \
    task question_answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
```

To make a request to your server, use the `requests` library and pass the request URL:

```python
import requests

url = "http://localhost:5543/predict"

obj = {
    "question": "Who is Mark?", 
    "context": "Mark is batman."
}

response = requests.post(url, json=obj)
```

In addition, you can make a request with a `curl` command from terminal:

```bash
curl -X POST \
  'http://localhost:5543/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Who is Mark?",
  "context": "Mark is batman."
}'
```
__ __
### Multiple Model Inference
To serve multiple models you can build a `config.yaml` file. 
In the sample YAML file below, we are defining two BERT models to be served by the `deepsparse.server` for the **question answering** task:

```yaml
num_cores: 2
num_workers: 2
endpoints:
    - task: question_answering
      endpoint: /unpruned/predict
      model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
    - task: question_answering
      endpoint: /pruned/predict
      model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni
      batch_size: 2
```
You can now run the server with the config file path passed in the `--config_file` argument:

```bash
deepsparse.server config config.yaml
```

You can send requests to a specific model by appending the model's `alias` from the `config.yaml` to the end of the request url. For example, to call the second model, the alias would be `question_answering/pruned_quant`:

```python
import requests

url = "http://localhost:5543/predict/question_answering/pruned_quant"

obj = {
    "question": "Who is Mark?", 
    "context": "Mark is batman."
}

response = requests.post(url, json=obj)
```

💡 **PRO TIP** 💡: While your server is running, you can always use the awesome swagger UI that's built into FastAPI to view your model's pipeline `POST` routes.
The UI also enables you to easily make sample requests to your server.
All you need is to add `/docs` at the end of your host URL:

    localhost:5543/docs

![alt text](./img/swagger_ui.png)

