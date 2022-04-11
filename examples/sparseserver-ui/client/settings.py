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

from pipelineclient import MultiPipelineClient


class FeatureHandler:

    """
    Class with front-end streamlit content features.
    """

    tasks_desc = "Select task:"
    tasks = [
        "Question Answering",
    ]

    variants_desc = "Select your sparse model:"
    variants = {
        "12-Layer BERT Base, Not Sparsified 😢": MultiPipelineClient(
            model="question_answering/base"
        ),
        "12-Layer BERT, Quantized, 99% of Base Accuracy": MultiPipelineClient(
            model="question_answering/12l_pruned80_quant"
        ),
        "6-Layer BERT, Quantized, 96% of Base Accuracy": MultiPipelineClient(
            model="question_answering/quant6lagg96"
        ),
        "3-Layer BERT, Quantized, 89% of Base Accuracy": MultiPipelineClient(
            model="question_answering/quant3lagg89"
        ),
        # "12-Layer BERT, Quantized, 95% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/pruned_quant"
        # ),
        # "12-Layer BERT, Quantized, 99% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/quantmod"
        # ),
        # "12-Layer BERT, 98% of Base Accuracy ": MultiPipelineClient(
        #     model="question_answering/agg98"
        # ),
        # "12-Layer BERT, 94% of Base Accuracy ": MultiPipelineClient(
        #     model="question_answering/agg94"
        # ),
        # "12-Layer BERT, 100% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/conserv"
        # ),
        # "6-Layer BERT, Quantized, 91% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/quant6lagg91"
        # ),
        # "6-Layer BERT, 98% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/6lagg98"
        # ),
        # "6-Layer BERT, 97% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/6lagg97"
        # ),
        # "6-Layer BERT, 96% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/6lagg96"
        # ),
        # "6-Layer BERT, 94% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/6lagg94"
        # ),
        # "3-Layer BERT, Quantized, 84% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/quant3lagg84"
        # ),
        # "3-Layer BERT, 90% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/3lagg90"
        # ),
        # "3-Layer BERT, 89% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/3lagg89"
        # ),
        # "3-Layer BERT, 86% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/3lagg86"
        # ),
        # "3-Layer BERT, 83% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/3lagg83"
        # ),
        # "12-Layer BERT, 90% of Base Accuracy": MultiPipelineClient(
        #     model="question_answering/12layer_pruned90"
        # ),
    }

    title = "<h1 style='text-align: Center; color: white;'>✨ Neural Magic ✨</h1>"
    subtitle = "<h2 style='text-align: Center; color: white; '>DeepSparse Server</h2>"

    code_banner = "Get started with faster inference 👇"
    code_text = "pip install deepsparse[server]"
    language = "python"
    repo_test = (
        "For code: [DeepSparse repo](https://github.com/neuralmagic/deepsparse)."
    )

    example_context_label = "Enter Context"
    example_question_label = "Enter Question"
    example_index_label = "Choose an example"
    example_index = ["example 1", "example 2", "example 3"]
    markdown_style = """
        <style>
        .big-font {
            font-size:40px !important;
        }
        </style>
        """
    footer = """
        <style>

        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        # background-color: #1816ff;
        background-color: blue;
        color: white;
        text-align: right;
        }
        </style>
        <div class="footer">
        <p>Made with ☠️ & Streamlit  .<a style="display: block; text-align: center;"</p>
        </div>
    """
