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

import time
from typing import Dict, List, Tuple, Union

import numpy

from deepsparse.utils import (
    get_input_names,
    get_output_names,
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

__all__ = ["ORTEngine"]

ARCH = cpu_architecture()
NUM_CORES = ARCH.num_available_physical_cores


def _validate_ort_import():
    if ort_import_error is not None:
        raise ImportError(
            "An exception occurred when importing onxxruntime. Please verify that "
            "onnxruntime is installed in order to use the onnxruntime inference "
            f"engine. \n\nException info: {ort_import_error}"
        )


def _validate_batch_size(batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    return batch_size


class ORTEngine(object):
    """
    Create a new ONNXRuntime Engine that compiles the given onnx file,

    Note 1: ORTEngines are compiled for a specific batch size and
    for a specific number of CPU cores.

    :param model: Either a path to the model's onnx file, a SparseZoo model stub
        prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
        object that defines the neural network
    :param batch_size: The batch size of the inputs to be used with the engine
    :param num_cores: The number of physical cores to run the model on.
    :param input_shapes: The list of shapes to set the inputs to. Pass None to use model as-is.
    """

    def __init__(
        self,
        model: Union[str, "Model", "File"],
        batch_size: int,
        num_cores: Union[None, int],
        input_shapes: List[List[int]] = None,
    ):
        _validate_ort_import()

        self._model_path = model_to_path(model)
        self._batch_size = _validate_batch_size(batch_size)
        self._num_cores = num_cores
        self._input_shapes = input_shapes

        self._input_names = get_input_names(self._model_path)
        self._output_names = get_output_names(self._model_path)

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3
        if num_cores is not None:
            sess_options.intra_op_num_threads = num_cores

        # TODO (michael): Unfortunately we are stacking overrides here, this can be
        # cleaned up once we pass the loaded ONNX around and not paths
        if self._input_shapes:
            with override_onnx_input_shapes(
                self._model_path, self._input_shapes
            ) as input_override_model_path:
                with override_onnx_batch_size(
                    input_override_model_path, batch_size
                ) as batch_override_model_path:
                    self._eng_net = onnxruntime.InferenceSession(
                        batch_override_model_path, sess_options
                    )
        else:
            with override_onnx_batch_size(
                self._model_path, batch_size
            ) as batch_override_model_path:
                self._eng_net = onnxruntime.InferenceSession(
                    batch_override_model_path, sess_options
                )

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
            self.__class__.__name__,
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
    def num_cores(self) -> int:
        """
        :return: The number of physical cores the current instance is running on
        """
        return self._num_cores if self._num_cores else NUM_CORES

    @property
    def scheduler(self) -> None:
        """
        :return: The kind of scheduler to execute with
        """
        return None

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
        if val_inp:
            self._validate_inputs(inp)
        inputs_dict = {name: value for name, value in zip(self._input_names, inp)}
        return self._eng_net.run(self._output_names, inputs_dict)

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
            if arr.shape[0] != self._batch_size:
                raise ValueError(
                    (
                        "array batch size of {} must match the batch size "
                        "the model was instantiated with {}"
                    ).format(arr.shape[0], self._batch_size)
                )

            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "array must be passed in as C contiguous, "
                    "call numpy.ascontiguousarray(array)"
                )

    def _properties_dict(self) -> Dict:
        return {
            "onnx_file_path": self.model_path,
            "batch_size": self.batch_size,
            "num_cores": self.num_cores,
        }
