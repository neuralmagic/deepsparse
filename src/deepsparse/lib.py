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

import importlib
import os


try:
    from deepsparse.cpu import cpu_details
except ImportError:
    raise ImportError(
        "Unable to import deepsparse python apis. "
        "Please contact support@neuralmagic.com"
    )

CORES_PER_SOCKET, AVX_TYPE, VNNI = cpu_details()


__all__ = [
    "get_neuralmagic_binaries_dir",
    "init_deepsparse_lib",
]


def get_neuralmagic_binaries_dir():
    nm_package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(nm_package_dir, AVX_TYPE)


def init_deepsparse_lib():
    try:
        onnxruntime_neuralmagic_so_path = os.path.join(
            get_neuralmagic_binaries_dir(), "deepsparse_engine.so"
        )
        spec = importlib.util.spec_from_file_location(
            "deepsparse.{}.deepsparse_engine".format(AVX_TYPE),
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
