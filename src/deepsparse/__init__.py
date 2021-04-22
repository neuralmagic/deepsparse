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
The DeepSparse package used to achieve GPU class performance
for Neural Networks on commodity CPUs.
"""

# flake8: noqa

from .cpu import (
    cpu_architecture,
    cpu_avx2_compatible,
    cpu_avx512_compatible,
    cpu_vnni_compatible,
)
from .engine import *
from .version import __version__
