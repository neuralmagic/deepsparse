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
from typing import Dict, List, Optional, Tuple, Union

import numpy


try:
    import torch

    torch_import_error = None
except Exception as torch_import_err:
    torch_import_error = torch_import_err
    torch = None

try:
    import onnxruntime

    ort_import_error = None
except Exception as ort_import_err:
    onnxruntime = None
    ort_import_error = ort_import_err

try:
    # flake8: noqa
    from deepsparse.cpu import cpu_architecture
except ImportError:
    raise ImportError(
        "Unable to import deepsparse python apis. "
        "Please contact support@neuralmagic.com"
    )

__all__ = ["TorchScriptEngine"]

_LOGGER = logging.getLogger(__name__)

ARCH = cpu_architecture()
NUM_CORES = ARCH.num_available_physical_cores


def _validate_torch_import():
    if ort_import_error is not None:
        raise ImportError(
            "An exception occurred when importing onxxruntime. Please verify that "
            "onnxruntime is installed in order to use the onnxruntime inference "
            "engine.\n\n`onnxruntime` can be installed by running the command "
            "`pip install deepsparse[onnxruntime]`"
            f"\n\nException info: {ort_import_error}"
        )


def _validate_batch_size(batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    return batch_size


def _select_device(device: str):
    device = str(device).lower()
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError(
            "Cuda not available is the local environment. Please select 'cpu'"
        )
    return "cpu"


def _validate_jit_model(model):
    if isinstance(model, torch.jit.ScriptModule):
        return
    raise ValueError(f"{model} is not a torch.jit model")


class TorchScriptEngine(object):
    """
    Given a loaded Torchscript(.pt) model or its saved file path, create an
     that compiles the given pytorch file,

    # Note 1: Engines are compiled for a specific batch size

    # :param model: Either a path to the model's .pt file or the loaded model
    # :param batch_size: The batch size of the inputs to be used with the engine
    # :param device: Hardware to run the engine on. Either cpu or cuda
    """

    def __init__(
        self,
        model: Union[str, "Model"],
        batch_size: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        if torch is None:
            raise ImportError(f"Unable to import torch, error: {torch_import_error}")

        _validate_torch_import()

        self._batch_size = _validate_batch_size(batch_size)
        self._device = _select_device(device)

        if isinstance(model, torch.nn.Module):
            self._model = model
        else:
            self._model = torch.jit.load(model).eval()
            _validate_jit_model(self._model)

        self._model.to(self.device)

    def __call__(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        """
        Convenience function for TorchScriptEngine.run(), see @run for more details

        | Example:
        |     engine = TorchScriptEngine("path/to/.pt", batch_size=1)
        |     inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        |     out = engine(inp)
        |     assert isinstance(out, List)

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for inference.
        :return: The list of outputs from the model after executing over the inputs
        """
        return self.run(inp, val_inp)

    def __repr__(self):
        """
        :return: Unambiguous representation of the current model instance
        """
        return "{}({})".format(self.__class__, self._properties_dict())

    def __str__(self):
        """
        :return: Human readable form of the current model instance
        """
        formatted_props = [
            "\t{}: {}".format(key, val) for key, val in self._properties_dict().items()
        ]

        return "{}:\n{}".format(
            self.__class__.__qualname__,
            "\n".join(formatted_props),
        )

    @property
    def model_path(self) -> str:
        """
        :return: The local path to the model file the current instance was compiled from
        """
        return self._model_path

    @property
    def batch_size(self) -> int:
        """
        :return: The batch size of the inputs to be used with the model
        """
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def scheduler(self) -> None:
        """
        :return: The kind of scheduler to execute with
        """
        return None

    @property
    def input_names(self) -> List[str]:
        """
        :return: The ordered names of the inputs.
        """
        return [node_arg.name for node_arg in self._eng_net.get_inputs()]

    @property
    def input_shapes(self) -> List[Tuple]:
        """
        :return: The ordered shapes of the inputs.
        """
        return [tuple(node_arg.shape) for node_arg in self._eng_net.get_inputs()]

    @property
    def output_names(self) -> List[str]:
        """
        :return: The ordered names of the outputs.
        """
        return [node_arg.name for node_arg in self._eng_net.get_outputs()]

    def run(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        """
        Run given inputs through the model for inference.
        Returns the result as a list of numpy arrays corresponding to
        the outputs of the model as defined in the ONNX graph.

        Note 1: the input dimensions must match what is defined in the torch.nn.Module.
        To avoid extra time in memory shuffles, the best use case
        is to format both the Module and the input into channels first format;
        ex: [batch, height, width, channels] => [batch, channels, height, width]

        Note 2: the input type for the numpy arrays must match
        what is defined in torch.nn.Module
        Generally float32 is most common,
        but int8 and int16 are used for certain layer and input types
        such as with quantized models.

        Note 3: the numpy arrays must be contiguous in memory,
        use numpy.ascontiguousarray(array) to fix if not.

        | Example:
        |     engine = TorchScriptEngine("path/to/.pt", batch_size=1, num_cores=None)
        |     inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        |     out = engine.run(inp)
        |     assert isinstance(out, List)

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for inference.
        :return: The list of outputs from the model after executing over the inputs
        """

        torch_inputs = [torch.from_numpy(input).to(self.device) for input in inp]

        tensors = self._model(*torch_inputs)

        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        return [tensor.cpu().detach().numpy() for tensor in tensors]

    def timed_run(
        self, inp: List[numpy.ndarray], val_inp: bool = False
    ) -> Tuple[List[numpy.ndarray], float]:
        """
        Convenience method for timing a model inference run.
        Returns the result as a tuple containing (the outputs from @run, time take)
        See @run for more details.


        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for inference.
        :return: The list of outputs from the model after executing over the inputs
        """
        start = time.perf_counter()
        out = self.run(inp, val_inp)
        end = time.perf_counter()

        return out, end - start

    def mapped_run(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> Dict[str, numpy.ndarray]:
        """
        Run given inputs through the model for inference.
        Returns the result as a dictionary of numpy arrays corresponding to
        the output names of the model as defined in the ONNX graph.

        Note 1: this function can add some a performance hit in certain cases.
        If using, please validate that you do not incur a performance hit
        by comparing with the regular run func

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for inference.
        :return: The dictionary of outputs from the model after executing
            over the inputs
        """
        out = self.run(inp, val_inp)
        return zip(self._output_names, out)

    def _properties_dict(self) -> Dict:
        return {
            "model_file_path": self.model_path,
            "batch_size": self.batch_size,
        }
