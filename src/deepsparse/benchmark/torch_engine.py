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

"""
Find out how ORTEngine is run in tests

Doc string


"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy

import torch
from deepsparse.utils import (
    model_to_path,
    override_onnx_batch_size,
    override_onnx_input_shapes,
)


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

# __all__ = ["TorchEngine"]

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


class TorchEngine(object):
    """
    # Create a new ONNXRuntime Engine that compiles the given onnx file,

    # Note 1: ORTEngines are compiled for a specific batch size and
    # for a specific number of CPU cores.

    # :param model: Either a path to the model's onnx file, a SparseZoo model stub
    #     prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
    #     object that defines the neural network
    # :param batch_size: The batch size of the inputs to be used with the engine
    # :param num_cores: The number of physical cores to run the model on.
    # :param input_shapes: The list of shapes to set the inputs to. Pass None to use model as-is.
    # :param providers: The list of execution providers executing with. Pass None to use all available.
    """

    def __init__(
        self,
        model: Union[str, "Model", "File"],  # pt file or Module pytorch
        batch_size: int = 1,
        # num_cores: Optional[int] = None,
        # input_shapes: Optional[List[List[int]]] = None,
        # providers: Optional[List[str]] = None,
    ):
        _validate_torch_import()

        if "state_dict" in model:
            self._model_path = None
        else:
            self._model_path = model_to_path(model)  # None if pass an actual Module

        self._batch_size = _validate_batch_size(batch_size)
        # self._num_cores = num_cores
        # self._input_shapes = input_shapes

        # if providers is None:
        #     providers = onnxruntime.get_available_providers()
        # self._providers = providers

        # torch.jit.save()
        

        self._model = (
            torch.jit.load(model) if isinstance(model, torch.nn.Module) else model
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            self._model.to(device)
            # add it to the data tensors


    def __call__(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        """
        Convenience function for ORTEngine.run(), see @run for more details

        | Example:
        |     engine = ORTEngine("path/to/onnx", batch_size=1, num_cores=None)
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
    # def num_cores(self) -> int:
    #     """
    #     :return: The number of physical cores the current instance is running on
    #     """
    #     return self._num_cores if self._num_cores else NUM_CORES

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

    @property
    def output_shapes(self) -> List[Tuple]:
        """
        :return: The ordered shapes of the outputs.
        """
        return [tuple(node_arg.shape) for node_arg in self._eng_net.get_outputs()]

    # @property
    # def providers(self) -> List[str]:
    #     """
    #     :return: The list of execution providers executing with
    #     """
    #     return self._providers

    def run(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        """
        Run given inputs through the model for inference.
        Returns the result as a list of numpy arrays corresponding to
        the outputs of the model as defined in the ONNX graph.

        Note 1: the input dimensions must match what is defined in the ONNX graph.
        To avoid extra time in memory shuffles, the best use case
        is to format both the onnx and the input into channels first format;
        ex: [batch, height, width, channels] => [batch, channels, height, width]

        Note 2: the input type for the numpy arrays must match
        what is defined in the ONNX graph.
        Generally float32 is most common,
        but int8 and int16 are used for certain layer and input types
        such as with quantized models.

        Note 3: the numpy arrays must be contiguous in memory,
        use numpy.ascontiguousarray(array) to fix if not.

        | Example:
        |     engine = ORTEngine("path/to/onnx", batch_size=1, num_cores=None)
        |     inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        |     out = engine.run(inp)
        |     assert isinstance(out, List)

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for inference.
        :return: The list of outputs from the model after executing over the inputs
        """
        print(0)
        torch_inputs = [torch.from_numpy(input) for input in inp]
        print(1)
        print()
        print()
        print(torch_inputs)
        print()
        print()
        print(self._model)
        print()
        print()
        print()
        # outputs = self._model(*torch_inputs)
        # self._model(tokens=inputs[0], attention_mask=inputs[1])

        # if val_inp:
        #     self._validate_inputs(inp)
        # inputs_dict = {name: value for name, value in zip(self.input_names, inp)}

        # return self._eng_net.run(self.output_names, inputs_dict)
        return self._model(*torch_inputs)

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

    def _validate_inputs(self, inp: List[numpy.ndarray]):
        if isinstance(inp, str) or not isinstance(inp, List):
            raise ValueError("inp must be a list, given {}".format(type(inp)))

        for arr in inp:

            # if arr.shape[0] != self._batch_size:
            #     raise ValueError(
            #         (
            #             "array batch size of {} must match the batch size "
            #             "the model was instantiated with {}"
            #         ).format(arr.shape[0], self._batch_size)
            #     )

            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "array must be passed in as C contiguous, "
                    "call numpy.ascontiguousarray(array)"
                )

    def _properties_dict(self) -> Dict:
        return {
            "onnx_file_path": self.model_path,
            "batch_size": self.batch_size,
            # "num_cores": self.num_cores,
            # "providers": self.providers,
        }
