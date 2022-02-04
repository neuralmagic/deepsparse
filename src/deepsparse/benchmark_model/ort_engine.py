import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy

from deepsparse.benchmark import BenchmarkResults
from deepsparse.utils import (
    model_to_path,
    get_input_names,
    get_output_names,
    override_onnx_input_shapes,
    override_onnx_batch_size,
)

try:
    import onnxruntime

    ort_import_error = None
except Exception as ort_import_err:
    onnxruntime = None
    ort_import_error = ort_import_err

try:
    from sparsezoo import Zoo
    from sparsezoo.objects import File, Model

    sparsezoo_import_error = None
except Exception as sparsezoo_err:
    Zoo = None
    Model = object
    File = object
    sparsezoo_import_error = sparsezoo_err

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
NUM_CORES = ARCH.num_physical_cores


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
    def __init__(
        self,
        model: Union[str, Model, File],
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

        with override_onnx_batch_size(
            self._model_path, batch_size
        ) as batch_override_model_path:
            if self._input_shapes:
                with override_onnx_input_shapes(
                    batch_override_model_path, self._input_shapes
                ) as input_override_model_path:
                    self._eng_net = onnxruntime.InferenceSession(
                        input_override_model_path, sess_options
                    )
            else:
                self._eng_net = onnxruntime.InferenceSession(
                    batch_override_model_path, sess_options
                )

    def __call__(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> List[numpy.ndarray]:
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

    def run(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        inputs_dict = {name: value for name, value in zip(self._input_names, inp)}
        return self._eng_net.run(self._output_names, inputs_dict)

    def timed_run(
        self, inp: List[numpy.ndarray], val_inp: bool = False
    ) -> Tuple[List[numpy.ndarray], float]:
        start = time.time()
        out = self.run(inp, val_inp)
        end = time.time()

        return out, end - start

    def _properties_dict(self) -> Dict:
        return {
            "onnx_file_path": self._model_path,
            "batch_size": self._batch_size,
            "num_cores": self._num_cores,
        }