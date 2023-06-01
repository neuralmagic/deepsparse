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
from deepsparse.analytics import deepsparse_analytics as _analytics


try:
    import cv2 as _cv2
    import openpifpaf as _openpifpaf
except ImportError:
    raise ImportError("Please install deepsparse[openpifpaf] to use this pathway")

from .utils import *


_analytics.send_event("python__open_pif_paf__init")
