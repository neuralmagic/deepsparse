"""
Code related to interfacing with a Neural Network in the DeepSparse Engine using python
"""

from typing import List, Dict, Optional, Iterable, Tuple
import os
import numpy
import importlib
import time

try:
    from deepsparse.cpu import cpu_details
    from deepsparse.version import *
except ImportError:
    raise ImportError(
        "Unable to import deepsparse python apis. "
        "Please contact support@neuralmagic.com"
    )


__all__ = ["Engine", "compile_model", "analyze_model"]


CORES_PER_SOCKET, AVX_TYPE, VNNI = cpu_details()


def _import_ort_nm():
    try:
        nm_package_dir = os.path.dirname(os.path.abspath(__file__))
        onnxruntime_neuralmagic_so_path = os.path.join(
            nm_package_dir, AVX_TYPE, "neuralmagic_onnxruntime_engine.so"
        )
        spec = importlib.util.spec_from_file_location(
            "deepsparse.{}.neuralmagic_onnxruntime_engine".format(AVX_TYPE),
            onnxruntime_neuralmagic_so_path,
        )
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)

        return engine
    except ImportError:
        raise ImportError(
            "Unable to import deepsparse engine binaries. "
            "Please contact support@neuralmagic.com"
        )


ENGINE_ORT = _import_ort_nm()


class Engine(object):
    """
    Create a new DeepSparse Engine that compiles the given onnx file
    for GPU class performance on commodity CPUs.

    Note 1: Engines are compiled for a specific batch size and
    for a specific number of CPU cores.

    Note 2: multi socket support is not yet built in to the Engine,
    all execution assumes single socket

    | Example:
    |    # create an engine for batch size 1 on all available cores
    |    engine = Engine("path/to/onnx", batch_size=1, num_cores=-1)

    :param onnx_file_path: The local path to the onnx file for the
        neural network graph definition to instantiate
    :param batch_size: The batch size of the inputs to be used with the engine
    :param num_cores: The number of physical cores to run the model on.
        Pass -1 to run on the max number of cores in one socket for the current machine
    """

    def __init__(self, onnx_file_path: str, batch_size: int, num_cores: int):
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
        Convenience function for Engine.run(), see @run for more details

        | Example:
        |     engine = Engine("path/to/onnx", batch_size=1, num_cores=-1)
        |     inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        |     out = engine(inp)
        |     assert isinstance(out, List)

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for the DeepSparse Engine
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

        return "{}.{}:\n{}".format(
            self.__class__.__module__,
            self.__class__.__name__,
            "\n".join(formatted_props),
        )

    @property
    def onnx_file_path(self) -> str:
        """
        :return: The local path to the onnx file the current instance was compiled from
        """
        return self._onnx_file_path

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
        return self._num_cores

    @property
    def num_sockets(self) -> int:
        """
        :return: The number of sockets the engine is compiled to run on;
            only current support is 1
        """
        return self._num_sockets

    @property
    def cpu_avx_type(self) -> str:
        """
        :return: The detected cpu avx type that neural magic is running with.
            One of {avx2, avx512}. AVX instructions give significant execution speedup
            with avx512 > avx2.
        """
        return self._cpu_avx_type

    @property
    def cpu_vnni(self) -> bool:
        """
        :return: True if vnni support was detected on the cpu, False otherwise.
            VNNI gives performance benefits for quantized networks.
        """
        return self._cpu_vnni

    def run(
        self, inp: List[numpy.ndarray], val_inp: bool = True,
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
        |     engine = Engine("path/to/onnx", batch_size=1, num_cores=-1)
        |     inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        |     out = engine.run(inp)
        |     assert isinstance(out, List)

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for the DeepSparse Engine
        :return: The list of outputs from the model after executing over the inputs
        """
        if val_inp:
            self._validate_inputs(inp)

        return self._eng_net.execute_list_out(inp)

    def timed_run(
        self, inp: List[numpy.ndarray], val_inp: bool = True
    ) -> Tuple[List[numpy.ndarray], float]:
        """
        Convenience method for timing a model inference run.
        Returns the result as a tuple containing (the outputs from @run, time take)

        See @run for more details.

        | Example:
        |     engine = Engine("path/to/onnx", batch_size=1, num_cores=-1)
        |     inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        |     out, time = engine.timed_run(inp)
        |     assert isinstance(out, List)
        |     assert isinstance(time, float)

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for the DeepSparse Engine
        :return: The list of outputs from the model after executing over the inputs
        """
        start = time.time()
        out = self.run(inp, val_inp)
        end = time.time()

        return out, end - start

    def mapped_run(
        self, inp: List[numpy.ndarray], val_inp: bool = True,
    ) -> Dict[str, numpy.ndarray]:
        """
        Run given inputs through the model for inference.
        Returns the result as a dictionary of numpy arrays corresponding to
        the output names of the model as defined in the ONNX graph.

        Note 1: this function can add some a performance hit in certain cases.
        If using, please validate that you do not incur a performance hit
        by comparing with the regular run func

        See @run for more details on specific setup for the inputs.

        | Example:
        |     engine = Engine("path/to/onnx", batch_size=1)
        |     inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        |     out = engine.mapped_run(inp)
        |     assert isinstance(out, Dict)

        :param inp: The list of inputs to pass to the engine for inference.
            The expected order is the inputs order as defined in the ONNX graph.
        :param val_inp: Validate the input to the model to ensure numpy array inputs
            are setup correctly for the DeepSparse Engine
        :return: The dictionary of outputs from the model after executing
            over the inputs
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
        A convenience function for quickly benchmarking the instantiated model
        on a give DataLoader in the DeepSparse Engine.
        batched_data must already shaped into the proper batch sizes
        for use with benchmarking.
        After executing, will return the summary statistics for benchmarking.

        :param batched_data: An iterator of input batches to be used for benchmarking.
        :param num_iterations: The number of iterations to run benchmarking for.
            Default is 20
        :param num_warmup_iterations: T number of iterations to warm up engine before
            benchmarking. These executions will not be counted in the benchmark
            results that are returned. Useful and recommended to bring
            the system to a steady state. Default is 5
        :param include_outputs: If True, outputs from forward passes during benchmarking
            will be returned under the 'outputs' key. Default is False
        :return: Dictionary of benchmark results including keys batch_stats_ms,
            batch_times_ms, and items_per_sec
        """
        assert num_iterations >= 1 and num_warmup_iterations >= 0, (
            "num_iterations and num_warmup_iterations must be non negative for "
            "benchmarking."
        )
        completed_iterations = 0
        batch_times = []
        outputs = []

        while completed_iterations < num_warmup_iterations + num_iterations:
            for batch in batched_data:
                # run benchmark
                output, batch_time = self.timed_run(batch, val_inp=False)

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
        A convenience function for quickly benchmarking the instantiated model
        on a given Dataset in the DeepSparse Engine.
        The data param must be individual items, the code will batch
        these items into the proper shape for the model for use with benchmarking.
        After executing, will return the summary statistics for benchmarking.

        :param data: An iterator of input items to be used for benchmarking.
            These items will be stacked to create batches of the proper batch_size.
            Items will be stacked in order. Will infinitely loop over the number
            of items to create the proper batch size and number of batches.
        :param num_iterations: The number of iterations to run benchmarking for.
            Default is 20
        :param num_warmup_iterations: T number of iterations to warm up engine before
            benchmarking. These executions will not be counted in the benchmark
            results that are returned. Useful and recommended to bring
            the system to a steady state. Default is 5
        :param include_outputs: If True, outputs from forward passes during benchmarking
            will be returned under the 'outputs' key. Default is False
        :return: Dictionary of benchmark results including keys batch_stats_ms,
            batch_times_ms, and items_per_sec
        """
        assert num_iterations >= 1 and num_warmup_iterations >= 0, (
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
            "onnx_file_path": self._onnx_file_path,
            "batch_size": self._batch_size,
            "num_cores": self._num_cores,
            "num_sockets": self._num_sockets,
            "cpu_avx_type": self._cpu_avx_type,
            "cpu_vnni": self._cpu_vnni,
        }


def compile_model(
    onnx_file_path: str, batch_size: int = 1, num_cores: int = -1
) -> Engine:
    """
    Convenience function to compile a model in the DeepSparse Engine
    from an ONNX file for inference.
    Gives defaults of batch_size == 1 and num_cores == -1
    (will use all physical cores available on a single socket).

    :param onnx_file_path: The local path to the onnx file for the
        neural network graph definition to instantiate
    :param batch_size: The batch size of the inputs to be used with the model
    :param num_cores: The number of physical cores to run the model on.
        Pass -1 to run on the max number of cores in one socket for the current machine
    :return: The created Engine after compiling the model
    """
    return Engine(onnx_file_path, batch_size, num_cores)


def analyze_model(
    onnx_file_path: str,
    input: List[numpy.ndarray],
    batch_size: int = 1,
    num_cores: int = -1,
    num_iterations: int = 20,
    num_warmup_iterations: int = 5,
    optimization_level: int = 1,
    imposed_as: Optional[float] = None,
    imposed_ks: Optional[float] = None,
) -> dict:
    """
    Function to analyze a model's performance in the DeepSparse Engine.
    The model must be defined in an ONNX graph and stored in a local file.
    Gives defaults of batch_size == 1 and num_cores == -1
    (will use all physical cores available on a single socket).

    :param onnx_file_path: The local path to the onnx file for the
        neural network graph definition to instantiate
    :param batch_size: The batch size of the inputs to be used with the model
    :param num_cores: The number of physical cores to run the model on.
        Pass -1 to run on the max number of cores in one socket for the current machine
    :param num_iterations: The number of times to repeat execution of the model
        while analyzing, default is 20
    :param num_warmup_iterations: The number of times to repeat execution of the model
        before analyzing, default is 5
    :param optimization_level: The amount of graph optimizations to perform.
        The current choices are either 0 (minimal) or 1 (all), default is 1
    :param imposed_as: Imposed activation sparsity, defaults to None.
        Will force the activation sparsity from all ReLu layers in the graph
        to match this desired sparsity level (percentage of 0's in the tensor).
        Beneficial for seeing how AS affects the performance of the model.
    :param imposed_ks: Imposed kernel sparsity, defaults to None.
        Will force all prunable layers in the graph to have weights with
        this desired sparsity level (percentage of 0's in the tensor).
        Beneficial for seeing how pruning affects the performance of the model.
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
