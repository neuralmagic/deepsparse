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
]


VALID_VECTOR_EXTENSIONS = {"avx2", "avx512"}


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
                "neuralmagic: architecture: can't modify {}".format(name)
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
        :return: the total number of hyperthreads available on the current machine
        """
        return self.threads_per_socket * self.num_sockets

    @property
    def num_physical_cores(self):
        """
        :return: the totla number of cores available on the current machine
        """
        return self.cores_per_socket * self.num_sockets


@_Memoize
def _parse_arch_bin() -> architecture:
    package_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(package_path, "arch.bin")

    try:
        info_str = subprocess.check_output(file_path).decode("utf-8")
        return architecture(json.loads(info_str))

    except Exception as ex:
        raise OSError(
            "neuralmagic: encountered exception while trying read arch.bin: {}".format(
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

    NM_ARCH environment variable can be used to override the avx instruction
    set detection

    :return: an instance of the architecture class
    """
    if not sys.platform.startswith("linux"):
        raise OSError("neuralmagic: only Linux platforms are supported.")

    arch = _parse_arch_bin()
    avx_type_override = os.getenv("NM_ARCH", None)

    if avx_type_override and avx_type_override != arch.isa:
        print(
            "neuralmagic: using env variable NM_ARCH={} for avx_type".format(
                avx_type_override
            )
        )
        if avx_type_override not in VALID_VECTOR_EXTENSIONS:
            raise OSError(
                (
                    "neuralmagic: invalid avx instruction set '{}' must be "
                    "one of {}."
                ).format(avx_type_override, ",".join(VALID_VECTOR_EXTENSIONS))
            )
        arch.override_isa(avx_type_override)

    if arch.isa not in VALID_VECTOR_EXTENSIONS:
        raise OSError(
            "neuralmagic: cannot determine avx instruction set. Set NM_ARCH to one of"
            " {} to continue.".format(",".join(VALID_VECTOR_EXTENSIONS))
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


def cpu_details() -> Tuple[int, str, bool]:
    """
    Detect the CPU details on linux systems
    If any other OS is used, will raise an exception

    Specifically:
        - the number of physical cores available per socket on the system
        - detects the vector instruction set available (avx2, avx512)
        - if vnni is available

    NM_ARCH environment variable can be used to override the avx instruction
    set detection

    :return: a tuple containing the detected cpu information
             (number of physical cores per socket, avx instruction set, vnni support)
    """
    arch = cpu_architecture()

    return arch.available_cores_per_socket, arch.isa, arch.vnni
