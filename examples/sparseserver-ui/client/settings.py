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

    variants_desc = "Select model:"
    variants = {
        "80% Pruned, Quantized for VNNI": MultiPipelineClient(
            model="question_answering/12l_pruned80_quant"
        ),
        "Pruned, Quant, 6layers, Aggressive 96": MultiPipelineClient(
            model="question_answering/quant6lagg96"
        ),
        "Pruned, Quant, 3layers, Aggressive 89": MultiPipelineClient(
            model="question_answering/quant3lagg89"
        ),
        "Base Model, Not Sparsified 😢": MultiPipelineClient(
            model="question_answering/base"
        ),
    }

    title = "<h1 style='text-align: Center; color: white;'>✨ Neural Magic ✨</h1>"
    subtitle = "<h2 style='text-align: Center; color: white; '> DeepSparse Server</h2>"

    code_banner = "Get started with faster inference 👇"
    code_text = "pip install deepsparse[server]"
    language = "python"
    repo_test = (
        "Give the [DeepSparse](https://github.com/neuralmagic/deepsparse) repo a ⭐!"
    )

    example_context_label = "Enter Context"
    example_question_label = "Enter Question"
    example_context = "The DeepSparse Engine is a CPU runtime that delivers \
    GPU-class performance by taking advantage of sparsity within neural \
    networks to reduce compute required as well as accelerate memory bound \
    workloads. It is focused on model deployment and scaling machine \
    learning pipelines, fitting seamlessly into your existing \
    deployments as an inference backend."
    example_question = (
        "What does the DeepSparse Engine take advantage of within neural networks?"
    )
    answer_label = "ANSWER: "
    time_label = "seconds"

    footer = """
        <style>

        .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        # background-color: #1816ff;
        background-color: #000000;
        color: white;
        text-align: right;
        }
        </style>
        <div class="footer">
        <p>Made with ☠️ & Streamlit  .<a style="display: block; text-align: center;"</p>
        </div>
    """
