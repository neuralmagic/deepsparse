"""
code related to interfacing with the Neural Magic inference engine using python
"""

from typing import List, Dict, Optional, Iterable
import os
import numpy
import importlib
import ctypes
import time
import warnings

try:
    from .cpu import cpu_details
    from .version import *
except ImportError:
    raise ImportError("Unable to import engine backend.")


__all__ = ["Model", "create_model", "analyze_model", "benchmark_model"]


CORES_PER_SOCKET, AVX_TYPE, VNNI = cpu_details()


def _import_ort_nm():
    try:
        nm_package_dir = os.path.dirname(os.path.abspath(__file__))
        onnxruntime_neuralmagic_so_path = os.path.join(
            nm_package_dir, AVX_TYPE, "neuralmagic_onnxruntime_engine.so"
        )
        spec = importlib.util.spec_from_file_location(
            "nmie.{}.neuralmagic_onnxruntime_engine".format(AVX_TYPE),
            onnxruntime_neuralmagic_so_path,
        )
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        return engine
    except ImportError:
        raise ImportError("Unable to import engine backend.")


ENGINE_ORT = _import_ort_nm()


class Model(object):
    """
    Python Model api for interfacing with the Neural Magic inference engine
    """

    def __init__(self, onnx_file_path: str, batch_size: int, num_cores: int):
        """
        Create a new model from the given onnx file in the Neural Magic engine for performance on CPUs.

        Note 1: models are created for a specific batch size and for a specific number of CPU cores.

        Note 2: multi socket support is not yet built in to the inference engine,
                all execution assumes single socket

        Example:
            # create model for batch size 1 on all available cores
            model = Model("path/to/onnx", batch_size=1, num_cores=-1)

        :param onnx_file_path: the local path to the onnx file for the neural network definition to instantiate
        :param batch_size: the batch size of the inputs to be used with the model
        :param num_cores: the number of physical cores to run the model on,
                          pass -1 to run on the max number of cores in one socket for current machine
        """

        if num_cores == -1:
            num_cores = CORES_PER_SOCKET

        if num_cores < 1:
            raise ValueError(
                "num_cores must be greater than 0: given {}".format(num_cores)
            )

        if not os.path.exists(onnx_file_path):
            raise ValueError(
                "onnx_file_path must exist: given {}".format(onnx_file_path)
            )

        self._onnx_file_path = onnx_file_path
        self._batch_size = batch_size
        self._num_cores = num_cores
        self._num_sockets = 1  # only single socket is supported currently
        self._cpu_avx_type = AVX_TYPE
        self._cpu_vnni = VNNI
        self._eng_net = ENGINE_ORT.neuralmagic_onnxruntime_engine(
            self._onnx_file_path, self._batch_size, self._num_cores, self._num_sockets
        )

    def __call__(
        self, inp: List[numpy.ndarray], val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        """
        Convenience function for model.forward(), see @forward for more details

        Example:
            model = Model("path/to/onnx", batch_size=1, num_cores=-1)
            inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
            out = model(inp)
            assert isinstance(out, List)

        :param inp: the list of inputs to pass to the model for inference
        :param val_inp: validate the input to the model to make sure the numpy array is setup correctly for the engine
        :return: the list of outputs from the model after executing over the inputs
        """
        return self.forward(inp, val_inp)

    def __repr__(self):
        """
        :return: unambiguous representation of the current model instance
        """
        return "{}({})".format(self.__class__, self._properties_dict())

    def __str__(self):
        """
        :return: human readable form of the current model instance
        """

        formatted_props = [
            "\t{}: {}".format(key, val) for key, val in self._properties_dict().items()
        ]

        return "{}.{}:\n{}".format(
            self.__class__.__module__,
            self.__class__.__name__,
            "\n".join(formatted_props),
        )

    @property
    def onnx_file_path(self) -> str:
        """
        :return: the local path to the onnx file the model was created from
        """
        return self._onnx_file_path

    @property
    def batch_size(self) -> int:
        """
        :return: the batch size of the inputs to be used with the model
        """
        return self._batch_size

    @property
    def num_cores(self) -> int:
        """
        :return: the number of physical cores to run the model on
        """
        return self._num_cores

    @property
    def num_sockets(self) -> int:
        """
        :return: the number of sockets the engine is compiled to run on, only current support is 1
        """
        return self._num_sockets

    @property
    def cpu_avx_type(self) -> str:
        """
        :return: the detected cpu avx type that neural magic is running with. One of {avx2, avx512}
        """
        return self._cpu_avx_type

    @property
    def cpu_vnni(self) -> bool:
        """
        :return: True if vnni support was detected on the cpu, False otherwise
        """
        return self._cpu_vnni

    def forward(
        self, inp: List[numpy.ndarray], val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        """
        Execute a forward pass through the model for the given input (inference)
        Returns the result as a list of numpy arrays corresponding to the outputs of the model as defined in the onnx.

        Note 1: the input dimensions must match what is given in the onnx.
        To avoid extra time in memory shuffles, best use case is to format both the onnx and the input into
        channels first format; ex: [batch, height, width, channels] => [batch, channels, height, width]

        Note 2: the input type for the numpy arrays must match what is given in the onnx.
        Generally float32 is most common, but int8 and int16 are used for certain layer and input types.

        Note 3: the numpy arrays must be contiguous in memory, use numpy.ascontiguousarray(array) to fix if not

        Example:
            model = Model("path/to/onnx", batch_size=1, num_cores=-1)
            inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
            out = model.forward(inp)
            assert isinstance(out, List)

        :param inp: the list of inputs to pass to the model for inference
        :param val_inp: validate the input to the model to make sure the numpy array is setup correctly for the engine.
                        Generally should be left at the default = True
        :return: the list of outputs from the model after executing over the inputs
        """
        if val_inp:
            self._validate_inputs(inp)

        return self._eng_net.execute_list_out(inp)

    def mapped_forward(
        self, inp: List[numpy.ndarray], val_inp: bool = True,
    ) -> Dict[str, numpy.ndarray]:
        """
        Execute a forward pass through the model for the given input (inference) and return a dictionary result.
        Each resulting tensor returned will be stored as a mapping in the dictionary.
        The keys are strings equal to the name as defined in onnx, the values are the output arrays.

        Note 1: this function can add some a performance hit in certain cases (it involves an extra memory copy)
        So, if using, please validate that you do not incur a performance hit by comparing with the regular forward func

        See @forward for more details on specific setup for the inputs.

        Example:
            model = Model("path/to/onnx", batch_size=1, num_cores=-1)
            inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
            out = model.mapped_forward(inp)
            assert isinstance(out, Dict)

        :param inp: the list of inputs to pass to the model for inference
        :param val_inp: validate the input to the model to make sure the numpy array is setup correctly for the engine.
                        Generally should be left at the default = True
        :return: the dictionary of outputs from the model after executing over the inputs
        """
        if val_inp:
            self._validate_inputs(inp)

        out = self._eng_net.execute(inp)

        return out

    def benchmark_batched(
        self,
        batched_data: Iterable[List[numpy.ndarray]],
        num_iterations: int = 20,
        num_warmup_iterations: int = 5,
        include_outputs: bool = False,
    ) -> Dict[str, float]:
        """
        :param batched_data: an iterator of input batches to be used for benchmarking.
        :param num_iterations: Number of iterations to run benchmarking for.
            Default is 20
        :param num_warmup_iterations: Number of iterations to warm up engine before
            benchmarking. Default is 5
        :param include_outputs: if True, outputs from forward passes during benchmarking
            will be returned under the 'outputs' key. Default is False
        :return: Dictionary of benchmark results including keys batch_stats_ms,
            batch_times_ms, and items_per_sec
        """
        assert num_iterations >= 0 and num_warmup_iterations >= 0, (
            "num_iterations and num_warmup_iterations must be non negative for "
            "benchmarking."
        )
        completed_iterations = 0
        batch_times = []
        outputs = []
        while completed_iterations < num_warmup_iterations + num_iterations:
            for batch in batched_data:
                # run benchmark
                batch_time = time.time()
                output = self.forward(batch)
                batch_time = time.time() - batch_time
                # update results
                batch_times.append(batch_time)
                if include_outputs:
                    outputs.append(output)
                # update loop
                completed_iterations += 1
                if completed_iterations >= num_warmup_iterations + num_iterations:
                    break
        batch_times = batch_times[num_warmup_iterations:]  # remove warmup times
        batch_times_ms = [batch_time * 1000 for batch_time in batch_times]
        items_per_sec = self.batch_size / numpy.mean(batch_times).item()

        batch_stats_ms = {
            "median": numpy.median(batch_times_ms),
            "mean": numpy.mean(batch_times_ms),
            "std": numpy.std(batch_times_ms),
        }

        benchmark_dict = {
            "batch_stats_ms": batch_stats_ms,
            "batch_times_ms": batch_times_ms,
            "items_per_sec": items_per_sec,
        }

        if include_outputs:
            benchmark_dict["outputs"] = outputs

        return benchmark_dict

    def benchmark(
        self,
        data: Iterable[List[numpy.ndarray]],
        num_iterations: int = 20,
        num_warmup_iterations: int = 5,
        include_outputs: bool = False,
    ) -> Dict[str, float]:
        """
        :param data: an iterator of input data to be used for benchmarking. Should
            be single data points with no batch dimension
        :param num_iterations: Number of iterations to run benchmarking for.
            Default is 20
        :param num_warmup_iterations: Number of iterations to warm up engine before
            benchmarking. Default is 5
        :param include_outputs: if True, outputs from forward passes during benchmarking
            will be returned under the 'outputs' key. Default is False
        :return: Dictionary of benchmark results including keys batch_stats_ms,
            batch_times_ms, and items_per_sec
        """
        assert num_iterations >= 0 and num_warmup_iterations >= 0, (
            "num_iterations and num_warmup_iterations must be non negative for "
            "benchmarking."
        )

        # define data loader
        def infinite_data_batcher():
            batch = []
            while True:
                for inputs in data:
                    batch.append(inputs)
                    if len(batch) == self.batch_size:
                        # concatenate batch inputs
                        batch_inputs = []
                        for input_idx in range(len(inputs)):
                            batch_input = [batch_val[input_idx] for batch_val in batch]
                            batch_inputs.append(numpy.stack(batch_input))
                        # yield and reset
                        yield batch_inputs
                        batch = []

        return self.benchmark_batched(
            batched_data=infinite_data_batcher(),
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            include_outputs=include_outputs,
        )

    def _validate_inputs(self, inp: List[numpy.ndarray]):
        if isinstance(inp, str) or not isinstance(inp, List):
            raise ValueError("inp must be a list, given {}".format(type(inp)))

        for arr in inp:
            if arr.shape[0] != self._batch_size:
                raise ValueError(
                    "array batch size of {} must match the batch size the model was instantiated with {}".format(
                        arr.shape[0], self._batch_size
                    )
                )

            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "array must be passed in as C contiguous, call numpy.ascontiguousarray(array)"
                )

    def _properties_dict(self) -> Dict:
        return {
            "onnx_file_path": self._onnx_file_path,
            "batch_size": self._batch_size,
            "num_cores": self._num_cores,
            "num_sockets": self._num_sockets,
            "cpu_avx_type": self._cpu_avx_type,
            "cpu_vnni": self._cpu_vnni,
        }


def create_model(
    onnx_file_path: str, batch_size: int = 1, num_cores: int = -1
) -> Model:
    """
    Convenience function to create a model in the Neural Magic engine from an onnx file for inference.
    Gives defaults of batch_size == 1 and num_cores == -1 (will use all physical cores available on a single socket)

    :param onnx_file_path: the local path to the onnx file for the neural network definition to instantiate
    :param batch_size: the batch size of the inputs to be used with the model, default is 1
    :param num_cores: the number of physical cores to run the model on, default is -1 (detect physical cores num)
    :return: the created model
    """
    return Model(onnx_file_path, batch_size, num_cores)


def analyze_model(
    onnx_file_path: str,
    input: List[numpy.ndarray],
    batch_size: int = 1,
    num_cores: int = -1,
    num_iterations: int = 1,
    num_warmup_iterations: int = 0,
    optimization_level: int = 1,
    imposed_as: Optional[float] = None,
    imposed_ks: Optional[float] = None,
) -> dict:
    """
    Function to analyze a model's performance using the Neural Magic engine from an onnx file for inference.
    Gives defaults of batch_size == 1 and num_cores == -1 (will use all physical cores available on a single socket)

    :param onnx_file_path: the local path to the onnx file for the neural network definition to instantiate
    :param input: the list of inputs to pass to the model for benchmarking
    :param batch_size: the batch size of the inputs to be used with the model, default is 1
    :param num_cores: the number of physical cores to run the model on, default is -1 (detect physical cores num)
    :param num_iterations: number of times to repeat execution, default is 1
    :param num_warmup_iterations: number of times to repeat unrecorded before starting actual benchmarking iterations
    :param optimization_level: how much optimization to perform?, default is 1
    :param imposed_as: imposed activation sparsity, defaults to None
    :param imposed_ks: imposed kernel sparsity, defaults to None

    :return: the analysis structure containing the performance details of each layer
    """
    if num_cores == -1:
        num_cores = CORES_PER_SOCKET
    num_sockets = 1  # only single socket is supported currently

    eng_net = ENGINE_ORT.neuralmagic_onnxruntime_engine(
        onnx_file_path, batch_size, num_cores, num_sockets
    )

    return eng_net.benchmark(
        input,
        num_iterations,
        num_warmup_iterations,
        optimization_level,
        imposed_as,
        imposed_ks,
    )


def benchmark_model(
    onnx_file_path: str,
    input: List[numpy.ndarray],
    batch_size: int = 1,
    num_cores: int = -1,
    num_iterations: int = 1,
    num_warmup_iterations: int = 0,
    optimization_level: int = 1,
    imposed_as: Optional[float] = None,
    imposed_ks: Optional[float] = None,
) -> dict:
    """
    DEPRECATED: Use nmie.analyze_model instead
    Function to analyze a model's performance using the Neural Magic engine from an onnx file for inference.
    Gives defaults of batch_size == 1 and num_cores == -1 (will use all physical cores available on a single socket)

    :param onnx_file_path: the local path to the onnx file for the neural network definition to instantiate
    :param input: the list of inputs to pass to the model for benchmarking
    :param batch_size: the batch size of the inputs to be used with the model, default is 1
    :param num_cores: the number of physical cores to run the model on, default is -1 (detect physical cores num)
    :param num_iterations: number of times to repeat execution, default is 1
    :param num_warmup_iterations: number of times to repeat unrecorded before starting actual benchmarking iterations
    :param optimization_level: how much optimization to perform?, default is 1
    :param imposed_as: imposed activation sparsity, defaults to None
    :param imposed_ks: imposed kernel sparsity, defaults to None

    :return: the analysis structure containing the performance details of each layer
    """
    warnings.warn(
        "Use of nmie.benchmark_model is deprecated. "
        "Use nmie.analyze_model instead."
    )
    return analyze_model(
        onnx_file_path,
        input,
        batch_size,
        num_cores,
        num_iterations,
        num_warmup_iterations,
        optimization_level,
        imposed_as,
        imposed_ks,
    )
