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

# flake8: noqa
from deepsparse.auto_install import auto_pip_install, Dependency

auto_pip_install(
    __qualname__,
    Dependency("torchvision", version=">=0.3.0,<=0.10.1"),
    optional_dependencies=[
        Dependency("click", version="<8.1"),
    ],
)


from .constants import *
from .pipelines import *
from .schemas import *

__all__ = constants.__all__ + pipelines.__all__ + schemas.__all__
