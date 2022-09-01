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
Functionality for detecting the details of the currently available cpu
"""

import json
import os
import subprocess
import sys
from typing import Any, Tuple


__all__ = [
    "VALID_VECTOR_EXTENSIONS",
    "architecture",
    "cpu_architecture",
    "cpu_details",
    "cpu_vnni_compatible",
    "cpu_avx2_compatible",
    "cpu_avx512_compatible",
    "cpu_neon_compatible",
    "cpu_sve_compatible",
    "cpu_quantization_compatible",
    "print_hardware_capability",
]


VALID_VECTOR_EXTENSIONS = {"avx2", "avx512", "neon", "sve"}


class _Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


class architecture(dict):
    """
    A class containing all the architecture details for the current CPU.

    Members include (but are not limited to):
        vendor                     - a string name of vendor)
        isa                        - a string containing avx2, avx512 or unknown)
        vnni                       - a boolean indicating VNNI support)
        num_sockets                - integer number of physical sockets
        available_sockets          - integer number of sockets available for use
        cores_per_socket           - integer number of physical cores per socket
        available_cores_per_socket - integer number of available cores per socket
        threads_per_core           - integer physical number of threads per core
        available_threads_per_core - integer available number of threads per core
        L1_instruction_cache_size  - L1 instruction cache size in bytes
        L1_data_cache_size         - L1 data cache size in bytes
        L2_cache_size              - L2 cache size in bytes
        L3_cache_size              - L3 cache size in bytes
    """

    def __init__(self, *args, **kwargs):
        super(architecture, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __setattr__(self, name: str, value: Any):
        if name != "__dict__":
            raise AttributeError(
                "Neural Magic: Architecture: can't modify {} to {}".format(name, value)
            )
        else:
            super(architecture, self).__setattr__(name, value)

    def override_isa(self, value: str):
        """
        Set the isa to the desired value.

        :param value: the value to update the isa to
        """
        object.__setattr__(self, "isa", value)

    @property
    def threads_per_socket(self):
        """
        :return: the number of hyperthreads available per socket on the current machine
        """
        return self.threads_per_core * self.cores_per_socket

    @property
    def num_threads(self):
        """
        :return: the total number of hyperthreads on the current machine
        """
        return self.threads_per_socket * self.num_sockets

    @property
    def num_physical_cores(self):
        """
        :return: the total number of cores on the current machine
        """
        return self.cores_per_socket * self.num_sockets

    @property
    def num_available_physical_cores(self):
        """
        :return: the total number of cores available on the current machine
        """
        return self.available_cores_per_socket * self.available_sockets


@_Memoize
def _parse_arch_bin() -> architecture:
    package_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(package_path, "arch.bin")

    try:
        info_str = subprocess.check_output(file_path).decode("utf-8")
        return architecture(json.loads(info_str))

    except subprocess.CalledProcessError as ex:
        error = json.loads(ex.stdout)
        raise OSError(
            "Neural Magic: Encountered exception while trying read arch.bin: {}".format(
                error["error"]
            )
        )

    except Exception as ex:
        raise OSError(
            "Neural Magic: Encountered exception while trying read arch.bin: {}".format(
                ex
            )
        )


def cpu_architecture() -> architecture:
    """
    Detect the CPU details on linux systems
    If any other OS is used, an exception will be raised.

    Specifically:
        - the number of physical cores available per socket on the system
        - detects the vector instruction set available (avx2, avx512)
        - if vnni is available

    NM_ARCH environment variable can be used to override the instruction
    set detection

    :return: an instance of the architecture class
    """
    if not sys.platform.startswith("linux"):
        raise OSError(
            "Neural Magic: Only Linux is supported, not '{}'.".format(sys.platform)
        )

    arch = _parse_arch_bin()
    isa_type_override = os.getenv("NM_ARCH", None)

    if isa_type_override and isa_type_override != arch.isa:
        print(
            "Neural Magic: Using env variable NM_ARCH={} for isa_type".format(
                isa_type_override
            )
        )
        if isa_type_override not in VALID_VECTOR_EXTENSIONS:
            raise OSError(
                (
                    "Neural Magic: Invalid instruction set '{}' must be " "one of {}."
                ).format(isa_type_override, ",".join(VALID_VECTOR_EXTENSIONS))
            )
        arch.override_isa(isa_type_override)

    if arch.isa not in VALID_VECTOR_EXTENSIONS:
        raise OSError(
            (
                "Neural Magic: Unable to determine instruction set '{}'. This system "
                "may be unsupported but to try, set NM_ARCH to one of {} to continue."
            ).format(arch.isa, ",".join(VALID_VECTOR_EXTENSIONS))
        )

    return arch


def cpu_vnni_compatible() -> bool:
    """
    :return: True if the current cpu has the VNNI instruction set,
        used for running int8 quantized networks performantly.
    """
    return cpu_architecture().vnni


def cpu_avx512_compatible() -> bool:
    """
    :return: True if the current cpu has the AVX512 instruction set,
        used for running neural networks performantly
    """
    return cpu_architecture().isa == "avx512"


def cpu_avx2_compatible() -> bool:
    """
    :return: True if the current cpu has the AVX2 or AVX512 instruction sets,
        used for running neural networks performantly
        (if AVX2 only then less performant compared to strictly AVX512)
    """
    return cpu_architecture().isa == "avx2" or cpu_avx512_compatible()


def cpu_neon_compatible() -> bool:
    """
    :return: True if the current cpu has the NEON instruction set,
        used for running neural networks performantly
    """
    return cpu_architecture().isa == "neon"


def cpu_sve_compatible() -> bool:
    """
    :return: True if the current cpu has the SVE instruction set,
        used for running neural networks performantly
    """
    return cpu_architecture().isa == "sve"


def cpu_quantization_compatible() -> bool:
    """
    :return: True if the current cpu has the AVX2, AVX512, NEON or SVE instruction sets,
        used for running quantized neural networks performantly.
        (AVX2 < AVX512 < VNNI)
    """
    return (
        cpu_avx2_compatible()
        or cpu_avx512_compatible()
        or cpu_neon_compatible()
        or cpu_sve_compatible()
    )


def cpu_details() -> Tuple[int, str, bool]:
    """
    Detect the CPU details on linux systems
    If any other OS is used, will raise an exception

    Specifically:
        - the number of physical cores available on the system
        - detects the vector instruction set available (avx2, avx512)
        - if vnni is available

    NM_ARCH environment variable can be used to override the avx instruction
    set detection

    :return: a tuple containing the detected cpu information
             (number of physical cores available, avx instruction set, vnni support)
    """
    arch = cpu_architecture()

    return arch.num_available_physical_cores, arch.isa, arch.vnni


def print_hardware_capability():
    """
    Print out the detected CPU's hardware capability and general support for
    model performance within the DeepSparse Engine.
    """
    arch = cpu_architecture()

    quantized_flag = "TRUE (emulated)" if cpu_quantization_compatible() else "FALSE"
    if cpu_vnni_compatible() or cpu_neon_compatible() or cpu_sve_compatible():
        quantized_flag = "TRUE"

    fp32_flag = (
        cpu_avx2_compatible()
        or cpu_avx512_compatible()
        or cpu_neon_compatible()
        or cpu_sve_compatible()
    )

    message = (
        f"{arch.vendor} CPU detected with {arch.num_available_physical_cores} cores. "
        f"({arch.available_sockets} sockets with "
        f"{arch.available_cores_per_socket} cores each)\n"
        f"DeepSparse FP32 model performance supported: {fp32_flag}.\n"
        "DeepSparse INT8 (quantized) model performance supported: "
        f"{quantized_flag}.\n\n"
    )

    if not (cpu_neon_compatible() or cpu_sve_compatible()):
        if cpu_avx2_compatible() and not cpu_avx512_compatible():
            message += (
                "AVX2 instruction set detected. Performance speedups are available, "
                "but inference time will be slower compared with an AVX-512 system.\n\n"
            )

        if cpu_quantization_compatible() and not cpu_vnni_compatible():
            message += (
                "Non VNNI system detected. Performance speedups for INT8 (quantized) "
                "models is available, but will be slower compared with a VNNI system. "
                "Set NM_FAST_VNNI_EMULATION=True in the environment to enable faster "
                "emulated inference which may have a minor effect on accuracy.\n\n"
            )

    message += f"Additional CPU info: {arch}"
    print(message)
