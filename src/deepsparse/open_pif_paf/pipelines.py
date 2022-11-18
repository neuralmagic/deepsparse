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
OpenPifPafPipeline
"""
from typing import Type

from deepsparse.open_pif_paf.schemas import OpenPifPafInput, OpenPifPafOutput
from deepsparse.pipeline import Pipeline


__all__ = [
    "OpenPifPafPipeline",
]


@Pipeline.register(
    task="open_pif_paf",
    default_model_path=("/home/damian/pifpaf/openpifpaf-resnet50.onnx"),
)
class OpenPifPafPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[OpenPifPafInput]:
        return OpenPifPafInput

    @property
    def output_schema(self) -> Type[OpenPifPafOutput]:
        return OpenPifPafOutput

    def setup_onnx_file_path(self) -> str:
        return self.model_path

    def process_inputs(self, inp):
        return inp.images

    def process_engine_outputs(self, out):
        return self.output_schema(out=out)
