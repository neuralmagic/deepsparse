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
Code related to interfacing with a Neural Network in the DeepSparse Engine using python
"""

import logging
import time
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy
from tqdm.auto import tqdm

from deepsparse.benchmark import BenchmarkResults
from deepsparse.utils import model_to_path, override_onnx_input_shapes


try:
    # flake8: noqa
    from deepsparse.cpu import cpu_architecture
    from deepsparse.lib import init_deepsparse_lib
    from deepsparse.version import *
except ImportError:
    raise ImportError(
        "Unable to import deepsparse python apis. "
        "Please contact support@neuralmagic.com"
    )


__all__ = [
    "Engine",
    "compile_model",
    "benchmark_model",
    "analyze_model",
    "Scheduler",
    "Context",
    "MultiModelEngine",
]

_LOGGER = logging.getLogger(__name__)

ARCH = cpu_architecture()
NUM_CORES = ARCH.num_available_physical_cores
# Default num-streams. Always 1 when using the single_stream scheduler. Depends on the
# hardware memory hierarchy when using elastic or multi_stream schedulers. It will
# typically be one per numa node, but can be higher on computers with multiple L3
# caches per numa node.
NUM_STREAMS = 0
AVX_TYPE = ARCH.isa
VNNI = ARCH.vnni

LIB = init_deepsparse_lib()


class Scheduler(Enum):
    """
    Scheduler kinds to determine Engine execution strategy.
    For most synchronous cases, the default single_stream is recommended.
    For running a model server or parallel inferences, try multi_stream for
    maximum utilization of hardware.

    - default: maps to single_stream
    - single_stream: requests from separate threads execute serially
    - multi_stream: requests from separate threads execute in parallel
    - elastic: requests from separate threads are distributed across NUMA nodes
    """

    default = "single_stream"
    single_stream = "single_stream"
    multi_stream = "multi_stream"
    elastic = "elastic"

    @staticmethod
    def from_str(key: str):
        if key in ("sync", "single", "single_stream"):
            return Scheduler.single_stream
        elif key in ("async", "multi", "multi_stream"):
            return Scheduler.multi_stream
        elif key in ("elastic"):
            return Scheduler.elastic
        else:
            raise ValueError(f"unsupported Scheduler: {key}")


def _validate_batch_size(batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    return batch_size


def _validate_num_cores(num_cores: Union[None, int]) -> int:
    if not num_cores:
        num_cores = NUM_CORES

    if num_cores < 1:
        raise ValueError("num_cores must be greater than 0")

    return num_cores


def _validate_num_streams(num_streams: Union[None, int], num_cores: int) -> int:
    if not num_streams:
        num_streams = NUM_STREAMS

    if num_streams < 0:
        raise ValueError("num_streams must be greater than 0")

    max_num_streams = NUM_CORES
    if max_num_streams > num_cores:
        max_num_streams = num_cores

    if num_streams > max_num_streams:
        num_streams = max_num_streams
        _LOGGER.warn(
            "num_streams exceeds num_cores - capping to {}".format(num_streams)
        )

    return num_streams


def _validate_scheduler(scheduler: Union[None, str, Scheduler]) -> Scheduler:
    if not scheduler:
        scheduler = Scheduler.default

    if isinstance(scheduler, str):
        scheduler = Scheduler.from_str(scheduler)

    if not isinstance(scheduler, Scheduler):
        raise ValueError(
            "unsupported type for scheduler: {} ({})".format(scheduler, type(scheduler))
        )

    return scheduler


class Engine(object):
    """
    Create a new DeepSparse Engine that compiles the given onnx file
    for GPU class performance on commodity CPUs.

    Note 1: Engines are compiled for a specific batch size and
    for a specific number of CPU cores.

    | Example:
    |    # create an engine for batch size 1 on all available cores
    |    engine = Engine("path/to/onnx", batch_size=1, num_cores=None)

    :param model: Either a path to the model's onnx file, a SparseZoo model stub
        prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
        object that defines the neural network
    :param batch_size: The batch size of the inputs to be used with the engine
    :param num_cores: The number of physical cores to run the model on. If more
        cores are requested than are available on a single socket, the engine
        will try to distribute them evenly across as few sockets as possible.
    :param num_streams: The max number of requests the model can handle
        concurrently.
    :param scheduler: The kind of scheduler to execute with. Pass None for the default.
    :param input_shapes: The list of shapes to set the inputs to. Pass None to use model as-is.
    """

    def __init__(
        self,
        model: Union[str, "Model", "File"],
        batch_size: int = 1,
        num_cores: int = None,
        num_streams: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
    ):
        self._model_path = model_to_path(model)
        self._batch_size = _validate_batch_size(batch_size)
        self._num_cores = _validate_num_cores(num_cores)
        self._scheduler = _validate_scheduler(scheduler)
        self._input_shapes = input_shapes
        self._cpu_avx_type = AVX_TYPE
        self._cpu_vnni = VNNI

        num_streams = _validate_num_streams(num_streams, self._num_cores)
        if self._input_shapes:
            with override_onnx_input_shapes(
                self._model_path, self._input_shapes
            ) as model_path:
                self._eng_net = LIB.deepsparse_engine(
                    model_path,
                    self._batch_size,
                    self._num_cores,
                    num_streams,
                    self._scheduler.value,
                    None,
                )
        else:
            self._eng_net = LIB.deepsparse_engine(
                self._model_path,
                self._batch_size,
                self._num_cores,
                num_streams,
                self._scheduler.value,
                None,
            )

    def __call__(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> List[numpy.ndarray]:
        """
        Convenience function for Engine.run(), see @run for more details

        | Example:
        |     engine = Engine("path/to/onnx", batch_size=1, num_cores=None)
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
    def num_cores(self) -> int:
        """
        :return: The number of physical cores the current instance is running on
        """
        return self._num_cores

    @property
    def num_streams(self) -> int:
        """
        :return: The max count of streams the current instance can handle
           concurrently.
        """
        return self._eng_net.num_streams()

    @property
    def scheduler(self) -> Scheduler:
        """
        :return: The kind of scheduler to execute with
        """
        return self._scheduler

    @property
    def fraction_of_supported_ops(self) -> float:
        """
        :return: The portion of the network supported by optimized runtime.
        """
        return round(self._eng_net.fraction_of_supported_ops(), 4)

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
        |     engine = Engine("path/to/onnx", batch_size=1, num_cores=None)
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
        self, inp: List[numpy.ndarray], val_inp: bool = False
    ) -> Tuple[List[numpy.ndarray], float]:
        """
        Convenience method for timing a model inference run.
        Returns the result as a tuple containing (the outputs from @run, time take)

        See @run for more details.

        | Example:
        |     engine = Engine("path/to/onnx", batch_size=1, num_cores=None)
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

    def benchmark(
        self,
        inp: List[numpy.ndarray],
        num_iterations: int = 20,
        num_warmup_iterations: int = 5,
        include_inputs: bool = False,
        include_outputs: bool = False,
        show_progress: bool = False,
    ) -> BenchmarkResults:
        """
        A convenience function for quickly benchmarking the instantiated model
        on a given input in the DeepSparse Engine.
        After executing, will return the summary statistics for benchmarking.

        :param inp: The list of inputs to pass to the engine for benchmarking.
            The expected order is the inputs order as defined in the ONNX graph.
        :param num_iterations: The number of iterations to run benchmarking for.
            Default is 20
        :param num_warmup_iterations: T number of iterations to warm up engine before
            benchmarking. These executions will not be counted in the benchmark
            results that are returned. Useful and recommended to bring
            the system to a steady state. Default is 5
        :param include_inputs: If True, inputs from forward passes during benchmarking
            will be added to the results. Default is False
        :param include_outputs: If True, outputs from forward passes during benchmarking
            will be added to the results. Default is False
        :param show_progress: If True, will display a progress bar. Default is False
        :return: the results of benchmarking
        """
        # define data loader
        def _infinite_loader():
            while True:
                yield inp

        return self.benchmark_loader(
            loader=_infinite_loader(),
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            include_inputs=include_inputs,
            include_outputs=include_outputs,
            show_progress=show_progress,
        )

    def benchmark_loader(
        self,
        loader: Iterable[List[numpy.ndarray]],
        num_iterations: int = 20,
        num_warmup_iterations: int = 5,
        include_inputs: bool = False,
        include_outputs: bool = False,
        show_progress: bool = False,
    ) -> BenchmarkResults:
        """
        A convenience function for quickly benchmarking the instantiated model
        on a give DataLoader in the DeepSparse Engine.
        After executing, will return the summary statistics for benchmarking.

        :param loader: An iterator of inputs to pass to the engine for benchmarking.
            The expected order of each input is as defined in the ONNX graph.
        :param num_iterations: The number of iterations to run benchmarking for.
            Default is 20
        :param num_warmup_iterations: T number of iterations to warm up engine before
            benchmarking. These executions will not be counted in the benchmark
            results that are returned. Useful and recommended to bring
            the system to a steady state. Default is 5
        :param include_inputs: If True, inputs from forward passes during benchmarking
            will be added to the results. Default is False
        :param include_outputs: If True, outputs from forward passes during benchmarking
            will be added to the results. Default is False
        :param show_progress: If True, will display a progress bar. Default is False
        :return: the results of benchmarking
        """
        assert num_iterations >= 1 and num_warmup_iterations >= 0, (
            "num_iterations and num_warmup_iterations must be non negative for "
            "benchmarking."
        )
        completed_iterations = 0
        results = BenchmarkResults()

        if show_progress:
            progress_bar = tqdm(total=num_iterations)

        while completed_iterations < num_warmup_iterations + num_iterations:
            for batch in loader:
                # run benchmark
                start = time.perf_counter()
                out = self.run(batch)
                end = time.perf_counter()

                if completed_iterations >= num_warmup_iterations:
                    # update results if warmup iterations are completed
                    results.append_batch(
                        time_start=start,
                        time_end=end,
                        batch_size=self.batch_size,
                        inputs=batch if include_inputs else None,
                        outputs=out if include_outputs else None,
                    )
                    if show_progress:
                        progress_bar.update(1)

                completed_iterations += 1

                if completed_iterations >= num_warmup_iterations + num_iterations:
                    break

        if show_progress:
            progress_bar.close()

        return results

    def analyze(
        self,
        inp: List[numpy.ndarray],
        num_iterations: int = 20,
        num_warmup_iterations: int = 5,
        optimization_level: int = 1,
        imposed_as: Optional[float] = None,
        imposed_ks: Optional[float] = None,
    ):
        """
        Function to analyze a model's performance in the DeepSparse Engine.

        Note 1: Analysis is currently only supported on a single socket.

        :param inp: The list of inputs to pass to the engine for analyzing inference.
            The expected order is the inputs order as defined in the ONNX graph.
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
        return self._eng_net.benchmark(
            inp,
            num_iterations,
            num_warmup_iterations,
            optimization_level,
            imposed_as,
            imposed_ks,
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
            "onnx_file_path": self.model_path,
            "batch_size": self.batch_size,
            "num_cores": self.num_cores,
            "num_streams": self.num_streams,
            "scheduler": self.scheduler,
            "fraction_of_supported_ops": self.fraction_of_supported_ops,
            "cpu_avx_type": self.cpu_avx_type,
            "cpu_vnni": self.cpu_vnni,
        }


class Context(object):
    """
    Contexts can be used to run multiple instances of the MultiModelEngine with the same
    scheduler. This allows one scheduler to manage the resources of the system
    effectively, keeping engines that are running different models from fighting over system
    resources.

    :param num_cores: The number of physical cores to run the model on. If more
        cores are requested than are available on a single socket, the engine
        will try to distribute them evenly across as few sockets as possible.
    :param num_streams: The max number of requests the model can handle
        concurrently.
    """

    def __init__(
        self,
        num_cores: int = None,
        num_streams: int = None,
    ):
        self._num_cores = _validate_num_cores(num_cores)
        self._scheduler = Scheduler.from_str("elastic")
        self._deepsparse_context = LIB.deepsparse_context(
            self._num_cores,
            _validate_num_streams(num_streams, self._num_cores),
            self._scheduler.value,
        )

    @property
    def value(self):
        return self._deepsparse_context

    @property
    def num_cores(self):
        return self._num_cores

    @property
    def num_streams(self):
        return self._deepsparse_context.num_streams()

    @property
    def scheduler(self):
        return self._scheduler

    def __repr__(self) -> str:
        return f"Context(num_cores={self.num_cores}, num_streams={self.num_streams}, scheduler={self.scheduler})"


class MultiModelEngine(Engine):
    """
    The MultiModelEngine, together with the Context class, can be used to run multiple models
    on the same computer at once. The interface and behavior are both very similar to the Engine
    class. The main difference is instead of taking in a scheduler and a number of cores as
    arguments to the constructor, the MultiModelEngine takes in a Context. The Context contains
    a shared scheduler along with other runtime information that will be used across instances
    of the MultiModelEngine to provide optimal performance when running multiple models
    concurrently.

    :param model: Either a path to the model's onnx file, a SparseZoo model stub
        prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
        object that defines the neural network
    :param batch_size: The batch size of the inputs to be used with the engine
    :param context: See above. This object should be constructed with the desired number of
        cores and passed into each instance of the MultiModelEngine.
    :param input_shapes: The list of shapes to set the inputs to. Pass None to use model as-is.
    """

    def __init__(
        self,
        model: Union[str, "Model", "File"],
        batch_size: int,
        context: Context,
        input_shapes: List[List[int]] = None,
    ):
        self._model_path = model_to_path(model)
        self._batch_size = _validate_batch_size(batch_size)
        self._num_cores = context.num_cores
        self._num_streams = context.num_streams
        self._scheduler = _validate_scheduler(context.scheduler)
        self._input_shapes = input_shapes
        self._cpu_avx_type = AVX_TYPE
        self._cpu_vnni = VNNI

        if self._input_shapes:
            with override_onnx_input_shapes(
                self._model_path, self._input_shapes
            ) as model_path:
                self._eng_net = LIB.deepsparse_engine(
                    model_path,
                    self._batch_size,
                    self._num_cores,
                    self._num_streams,
                    self._scheduler.value,
                    context.value,
                )
        else:
            self._eng_net = LIB.deepsparse_engine(
                self._model_path,
                self._batch_size,
                self._num_cores,
                self._num_streams,
                self._scheduler.value,
                context.value,
            )


def compile_model(
    model: Union[str, "Model", "File"],
    batch_size: int = 1,
    num_cores: int = None,
    num_streams: int = None,
    scheduler: Scheduler = None,
    input_shapes: List[List[int]] = None,
) -> Engine:
    """
    Convenience function to compile a model in the DeepSparse Engine
    from an ONNX file for inference.
    Gives defaults of batch_size == 1, num_cores == None
    (will use all physical cores available on the system).

    :param model: Either a path to the model's onnx file, a SparseZoo model stub
        prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
        object that defines the neural network
    :param batch_size: The batch size of the inputs to be used with the model
    :param num_cores: The number of physical cores to run the model on.
        Pass None or 0 to run on the max number of cores for the current
        machine; default None
    :param num_streams: The max number of requests the model can handle
        concurrently. None or 0 implies a scheduler-defined default value;
        default None
    :param scheduler: The kind of scheduler to execute with. Pass None for the default.
    :param input_shapes: The list of shapes to set the inputs to. Pass None to use model as-is.
    :return: The created Engine after compiling the model
    """
    return Engine(
        model=model,
        batch_size=batch_size,
        num_cores=num_cores,
        num_streams=num_streams,
        scheduler=scheduler,
        input_shapes=input_shapes,
    )


def benchmark_model(
    model: Union[str, "Model", "File"],
    inp: List[numpy.ndarray],
    batch_size: int = 1,
    num_cores: int = None,
    num_streams: int = None,
    num_iterations: int = 20,
    num_warmup_iterations: int = 5,
    include_inputs: bool = False,
    include_outputs: bool = False,
    show_progress: bool = False,
    scheduler: Scheduler = None,
    input_shapes: List[List[int]] = None,
) -> BenchmarkResults:
    """
    Convenience function to benchmark a model in the DeepSparse Engine
    from an ONNX file for inference.
    Gives defaults of batch_size == 1 and num_cores == None
    (will use all physical cores available on a single socket).

    Note 1: Benchmarking is currently only supported on a single socket.

    :param model: Either a path to the model's onnx file, a SparseZoo model stub
        prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
        object that defines the neural network
    :param batch_size: The batch size of the inputs to be used with the model
    :param num_cores: The number of physical cores to run the model on.
        Pass None or 0 to run on the max number of cores
        for the current machine; default None
    :param num_streams: The max number of requests the model can handle
        concurrently. None or 0 implies a scheduler-defined default value;
        default None
    :param inp: The list of inputs to pass to the engine for benchmarking.
        The expected order is the inputs order as defined in the ONNX graph.
    :param num_iterations: The number of iterations to run benchmarking for.
        Default is 20
    :param num_warmup_iterations: T number of iterations to warm up engine before
        benchmarking. These executions will not be counted in the benchmark
        results that are returned. Useful and recommended to bring
        the system to a steady state. Default is 5
    :param include_inputs: If True, inputs from forward passes during benchmarking
        will be added to the results. Default is False
    :param include_outputs: If True, outputs from forward passes during benchmarking
        will be added to the results. Default is False
    :param show_progress: If True, will display a progress bar. Default is False
    :param scheduler: The kind of scheduler to execute with. Pass None for the default.
    :return: the results of benchmarking
    """
    model = compile_model(
        model=model,
        batch_size=batch_size,
        num_cores=num_cores,
        num_streams=num_streams,
        scheduler=scheduler,
        input_shapes=input_shapes,
    )

    return model.benchmark(
        inp,
        num_iterations,
        num_warmup_iterations,
        include_inputs,
        include_outputs,
        show_progress,
    )


def analyze_model(
    model: Union[str, "Model", "File"],
    inp: List[numpy.ndarray],
    batch_size: int = 1,
    num_cores: int = None,
    num_iterations: int = 20,
    num_warmup_iterations: int = 5,
    optimization_level: int = 1,
    imposed_as: Optional[float] = None,
    imposed_ks: Optional[float] = None,
    scheduler: Scheduler = None,
    input_shapes: List[List[int]] = None,
) -> dict:
    """
    Function to analyze a model's performance in the DeepSparse Engine.
    The model must be defined in an ONNX graph and stored in a local file.
    Gives default batch_size == 1 and num_cores == None
    (will use all physical cores available on a single socket).

    Note 1: Analysis is currently only supported on a single socket.

    :param model: Either a path to the model's onnx file, a SparseZoo model stub
        prefixed by 'zoo:', a SparseZoo Model object, or a SparseZoo ONNX File
        object that defines the neural network graph definition to analyze
    :param inp: The list of inputs to pass to the engine for analyzing inference.
        The expected order is the inputs order as defined in the ONNX graph.
    :param batch_size: The batch size of the inputs to be used with the model
    :param num_cores: The number of physical cores to run the model on.
        Pass None or 0 to run on the max number of cores
        for the current machine; default None
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
    :param scheduler: The kind of scheduler to execute with. Pass None for the default.
    :return: the analysis structure containing the performance details of each layer
    """
    model = compile_model(
        model=model,
        batch_size=batch_size,
        num_cores=num_cores,
        scheduler=scheduler,
        input_shapes=input_shapes,
    )

    return model.analyze(
        inp,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        optimization_level=optimization_level,
        imposed_as=imposed_as,
        imposed_ks=imposed_ks,
    )
