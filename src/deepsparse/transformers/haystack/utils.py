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

from haystack.utils import print_documents

from deepsparse.transformers.haystack import HaystackPipelineOutput


__all__ = [
    "print_pipeline_documents",
]


def print_pipeline_documents(
    haystack_pipeline_output: HaystackPipelineOutput,
) -> None:
    """
    Helper function to print documents directly from NM Haystack Pipeline outputs

    :param haystack_pipeline_output: instance of HaystackPipelineOutput schema
    :return: None
    """
    if isinstance(haystack_pipeline_output.query, list):
        for i in range(len(haystack_pipeline_output.query)):
            results_dict = {
                key: value[i] for key, value in haystack_pipeline_output.dict().items()
            }
            print_documents(results_dict)
    else:
        print_documents(haystack_pipeline_output.dict())
