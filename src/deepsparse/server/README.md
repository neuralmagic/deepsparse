### DeepSparse Server 🔌

The DeepSparse inference server allows you to serve models and pipelines in deployment in HTTP. The server runs on top of the popular FastAPI web framework and Uvicorn web server. Currently, the server only supports NLP tasks, however support for computer vision will soon be released in upcoming versions!

 - run `deepsparse.server --help` to lookup the available CLI arguments.


        --host TEXT                     Bind socket to this host. Use --host 0.0.0.0
                                        to make the application available on your
                                        local network. IPv6 addresses are supported,
                                        for example: --host '::'. Defaults to
                                        0.0.0.0

        --port INTEGER                  Bind to a socket with this port. Defaults to
                                        5543.

        --workers INTEGER               Use multiple worker processes. Defaults to
                                        1.

        --log_level                     [debug|info|warn|critical|fatal]
                                        Bind to a socket with this port. Defaults to
                                        info.

        --config_file TEXT              Configuration file containing info on how to
                                        serve the desired models.

        --task TEXT                     The task the model_path is serving. For
                                        example, one of: question_answering,
                                        text_classification, token_classification.

                                        Ignored if config file is supplied
        --model_path TEXT               The path to a model.onnx file, a model
                                        folder containing the model.onnx and
                                        supporting files, or a SparseZoo model stub.
                                        Ignored if config_file is supplied.

        --batch_size INTEGER            The batch size to serve the model from
                                        model_path with. Ignored if config_file is
                                        supplied.

        --help                          Show this message and exit.


##### Single Model Inference

Example CLI command for serving a single model:

```bash
deepsparse.server \
    --task question_answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"
```

##### Multiple Model Inference
To serve multiple models you can easily build a `config.yaml` file. 
In the sample yaml below, we are defining 2 BERT models to be served by the `deepsparse.server` for the question answering task:

    models:
    - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
        batch_size: 1
        alias: question_answering/dense
    - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95
        batch_size: 1
        alias: question_answering/sparse_quantized

After you finish building the `config.yaml` file, you can run the server with the config file path passed in the `--config_file` argument:
```bash
deepsparse.server --config_file config.yaml
```

💡 PRO-TIP 💡: When your server is running, you can always use the awesome swagger UI that's built in to FastAPI to view your model's pipeline `POST` routes. All you need is to add `/docs` at the end of your host URL:

    localhost:5543/docs

![alt text](./img/swagger_ui.png)
__ __
### Client 📲

When the DeepSparse HTTP server is up and running you can can send it requests via our PipelineClient object:


    import json
    from typing import List
    import numpy
    import requests

    class PipelineClient:
        """
        Client object for making requests to the example DeepSparse BERT inference server

        :param alias: model alias of FastAPI route
        :param address: IP address of the server, default is 0.0.0.0
        :param port: Port the server is hosted on, default is 5543
        """

        def __init__(self, alias: str, address: str ='0.0.0.0', port: str ='5543'):

            self.alias = alias
            self._url = f'http://{address}:{port}/predict/{self.alias}'
            
        def __call__(self, **kwargs) -> List[numpy.ndarray]:

            """
            :param kwargs: named inputs to the model server pipeline. e.g. for
                question-answering - `question="...", context="..."

            :return: json outputs from running the model server pipeline with the given
                input(s)
            """

            response = requests.post(self._url, json=kwargs)
            return json.loads(response.content)

💡 PRO-TIP 💡: the `config.yaml` file uses the `alias` variable to label your `/predict` routes on the server! 

For example, if you wanted to send a request to the second model in the `config.yaml` example shown above, here's how the PipelineClient object would be initialized to make a request:

    PipelineClient(alias='question_answering/sparse_quantized')