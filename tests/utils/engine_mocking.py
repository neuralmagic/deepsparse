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

import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, List, Type
from unittest import mock

import numpy
import onnx
import onnxruntime as ort


class SampleMode(str, Enum):
    """
    How to sample outputs generated from the mocked engine.
    """

    zeros = "zeros"
    rand = "rand"


def mock_engine(*, rng_seed: int, mode: SampleMode = SampleMode.rand, **kwargs):
    """
    Intended to create a fake engine instead of compiling the model.

    This patches the `LIB.deepsparse_engine()` that is used in deepsparse.Engine
    with `_FakeDeepsparseLibEngine`.

    Use like you would a regular `unittest.mock.patch`
    ```python
    @mock_engine(rng_seed=0)
    def test_something(engine):
        ...
    ```

    If you want the arrays to be zeros instead of random data:
    ```python
    @mock_engine(rng_seed=0, model=SampleMode.zeros)
    def test_something(engine):
        ...
    ```
    """

    def deepsparse_engine(*args, **kwargs):
        return _FakeDeepsparseLibEngine(*args, **kwargs, rng_seed=rng_seed, mode=mode)

    return mock.patch(
        "deepsparse.engine.LIB.deepsparse_engine",
        side_effect=deepsparse_engine,
        **kwargs,
    )


class _FakeDeepsparseLibEngine:
    def __init__(
        self,
        model_path: str,
        batch_size: int,
        num_cores: int,
        num_streams: int,
        scheduler_value: Any,
        context_value: Any,
        *,
        # NOTE: the following are not actual deepsparse engine arguments
        rng_seed: int,
        mode: SampleMode,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_cores = num_cores
        self.num_streams = num_streams
        self.scheduler_value = scheduler_value
        self.context_value = context_value
        self.rng_seed = rng_seed
        self.mode = mode

        # override batch dimension in inputs & outputs
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = onnx.load(self.model_path)
            inputs = model.graph.input
            initializer_names = [node.name for node in model.graph.initializer]
            # remove initializer nodes from inputs
            inputs = [input for input in inputs if input.name not in initializer_names]

            # This is done because not all `.onnx` files passed in will
            # conform to expected `batch_size`.
            #
            # By overriding the batch size in the inputs, and loading this
            # model with the onnxruntime, onnxruntime will automatically
            # fix the batch dimension in outputs.
            #
            # Assumes the first dimension is batch dimension!!
            # However in general we cannot assume that all outputs have
            # a batch dimension, that's why we need onnxruntime here.
            for input in inputs:
                input.type.tensor_type.shape.dim[0].dim_value = batch_size

            batched_model_path = str(Path(tmp_dir) / "batched_model.onnx")
            onnx.save(model, batched_model_path)

            session = ort.InferenceSession(batched_model_path)
            self.input_descriptors = list(map(_to_descriptor, session.get_inputs()))
            self.output_descriptors = list(map(_to_descriptor, session.get_outputs()))

    def execute(self, inputs):
        raise NotImplementedError("mapped_run not supported with mocked engine")

    def execute_list_out(self, inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        # validate all the inputs are the correct type
        assert len(inputs) == len(self.input_descriptors)
        for descriptor, input in zip(self.input_descriptors, inputs):
            descriptor.check(input)

        # NOTE: we construct the rng here because this function may
        # be called from different threads, and rng's are not
        # thread safe.
        rng = numpy.random.default_rng(self.rng_seed)

        outputs = [d.sample(rng, self.mode) for d in self.output_descriptors]

        assert len(outputs) == len(self.output_descriptors)
        for descriptor, output in zip(self.output_descriptors, outputs):
            descriptor.check(output)
        return outputs


def _to_descriptor(node: ort.NodeArg) -> "_NumpyDescriptor":
    to_numpy_dtype = {
        "tensor(float)": numpy.float32,
        "tensor(double)": numpy.float64,
        "tensor(uint8)": numpy.uint8,
        "tensor(int64)": numpy.int64,
    }
    return _NumpyDescriptor(shape=node.shape, dtype=to_numpy_dtype[node.type])


class _NumpyDescriptor:
    def __init__(self, *, shape: List[int], dtype: Type):
        self.shape = shape
        self.dtype = dtype

    def check(self, a: numpy.ndarray):
        assert isinstance(a, numpy.ndarray)
        assert a.shape == tuple(self.shape), f"Expected {self} found shape={a.shape}"
        assert a.dtype == self.dtype, f"Expected {self} found dtype={a.dtype}"

    def sample(self, rng: numpy.random.Generator, mode: SampleMode) -> numpy.ndarray:
        if mode == SampleMode.zeros:
            return numpy.zeros(self.shape, dtype=self.dtype)
        elif mode == SampleMode.rand:
            return rng.random(size=self.shape).astype(self.dtype)
        else:
            raise ValueError(f"invalid sample mode {mode}")

    def __repr__(self) -> str:
        return f"_NumpyDescriptor(shape={self.shape}, dtype={self.dtype})"
