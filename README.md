# Neural Magic Inference Engine

[TODO: mission statement] Inference engine for running neural networks efficiently and performantly

[TODO: chips for build info, license, website, release, etc]

[TODO: longer description and use cases]

[TODO: example screenshot]

[TODO: links out to other products]

## Quick Tour
[TODO: examples]

#### Dense MobilenetV1 Example
```python
from deepsparse import compile_model
from sparsezoo.models.classification import mobilenet_v1
​
model = mobilenet_v1()
engine = compile_model(model)
​
inputs = model.data_inputs.sample_batch()
outputs = engine.run(inputs)
```

#### Sparse MobilenetV1 Example

```python
from deepsparse import compile_model
from sparsezoo.models.classification import mobilenet_v1
​
model = mobilenet_v1(optim_name="sparse", optim_category="aggressive")
engine = compile_model(model)
​
inputs = model.data_inputs.sample_batch()
outputs = engine.run(inputs)
```

#### Benchmark MobilenetV1 Example

```python
from deepsparse import benchmark_model
from sparsezoo.models.classification import mobilenet_v1
​
batch_size = 64
model = mobilenet_v1(optim_name="sparse", optim_category="aggressive")
engine = compile_model(model, batch_size)
results = engine.benchmark(model.data_inputs.sample_batch(batch_size))
​print(results)
```

## Why should I use?
[TODO: use cases]

## Installation
[TODO: installation instructions]

## Development Setup
[TODO: dev instructions]

## Models and Recipes
[TODO: models and recipes table]

## Learn More
[TODO: table with links for deeper topics]

## Citation
[TODO: list out any citations]

## Release History
[TODO: release history]
