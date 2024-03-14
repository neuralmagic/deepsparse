# **Step 1: Installation**

Install DeepSparse and MLServer.

```bash
pip install -r requirements.txt
```

# **Step 2: Write Custom Runtime**

We need to write a [Custom Inference Runtime](https://mlserver.readthedocs.io/en/stable/user-guide/custom.html) to use DeepSparse within MLServer.

### Implement `load()` and `predict()`

First, we implement the `load()` and `predict()` methods in `models/text-classification-model/models.py`. Note that your implementation of the of `load()` and `predict()` will vary by the task that you choose.

Here's an example for text classification:
```python
from mlserver import MLModel
from mlserver.codecs import decode_args
from typing import List
from deepsparse import Pipeline

class DeepSparseRuntime(MLModel):
    async def load(self) -> bool:
        # compiles the pipeline
        self._pipeline = Pipeline.create(
            task = self._settings.parameters.task,                          # from model-settings.json
            model_path = self._settings.parameters.model_path,              # from model-settings.json
            batch_size = self._settings.parameters.batch_size,              # from model-settings.json
            sequence_length = self._settings.parameters.sequence_length,    # from model-settings.json
        )
        return True
    
    @decode_args
    async def predict(self, sequences: List[str]) -> List[str]:
        # runs the inference
        prediction = self._pipeline(sequences=sequences)
        return prediction.labels
```

### Create `model-settings.json`

Second, we create a config at `models/text-classification-model/model-settings.json`. In this file, we will specify the location of the implementation of the custom runtime as well as the 
paramters of the deepsparse inference session.

```json
{
    "name": "text-classification-model",
    "implementation": "models.DeepSparseRuntime",
    "parameters": {
        "task": "text-classification",
        "model_path": "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none",
        "batch_size": 1,
        "sequence_length": 128
    }
}
```

# **Step 3: Launch MLServer**

Launch the server with the CLI:

```bash
mlserver start ./models/text-classification-model/
```

# **Step 4: Send Inference Requests**

Now, an inference endpoint is exposed at `http://localhost:8080/v2/models/text-classification-model/infer`. `client.py` is a sample script for requesting the endpoint.

Run the following:
```python
python3 client.py
```