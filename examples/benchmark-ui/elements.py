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


class TextElements:

    # Markdown Text
    md_title = "# Benchmarking Sparse Models on DigitalOcean's CPUs!"

    md_body = """

    ### Hey There 👋
    
    Welcome to Neural Magic's Benchmarking Demo where you can put \
    [DeepSparse](https://github.com/neuralmagic/deepsparse) on \
    DigitalOcean's CPUs to the test! \
    Our goal is to provide an easy-to-use platform for users to \
    benchmark select models from Computer Vision and NLP domains, \
    and to do so with a variety of different configurations.

    Whether you're an expert in the field or just getting started, \
    our demo provides a straightforward way to load sparse deep learning \
    models into DeepSparse and get valuable insights into its performance on CPUs.
    To get started:
    1.  Select your AI task in the tabs.
    2.  Select your model. You have the choice of benchmarking a sparse or dense model.
    3.  Select your engine.
    4.  Set your batch size, which refers to the number of input samples that are \
        processed.
    5.  Set the time of how long the benchmarking will take.
    6.  Select your inference scenario.
        - SYNC is used to simulate model latency/synchronous scenarios.
        - ASYNC is used to simulate model throughput/asynchronous scenarios.
    """

    # Models
    image_detection_models = [
        "Sparse YOLOv5s COCO",
        "Dense YOLOv5s COCO",
        "Sparse YOLOv5m COCO",
        "Dense YOLOv5m COCO",
    ]
    image_classification_models = [
        "Sparse ResNet 50 ImageNet",
        "Dense ResNet 50 ImageNet",
        "Sparse MobileNetV1 ImageNet",
        "Dense MobileNetV1 ImageNet",
    ]
    image_segmentation_models = ["Sparse YOLACT COCO", "Dense YOLACT COCO"]
    sentiment_analysis_models = [
        "Sparse DistilBERT SST2",
        "Dense DistilBERT SST2",
        "Sparse oBERT SST2",
        "Dense oBERT SST2",
    ]
    question_answering_models = [
        "Sparse DistilBERT SQUAD",
        "Dense DistilBERT SQUAD",
        "Sparse oBERT SQUAD",
        "Dense oBERT SQUAD",
    ]
    token_classification_models = [
        "Sparse DistilBERT CONLL",
        "Dense DistilBERT CONLL",
        "Sparse oBERT CONLL",
        "Dense oBERT CONLL",
    ]
    document_classification_models = [
        "Sparse RoBERTa IMDB",
        "Dense RoBERTa IMDB",
        "Sparse oBERT IMDB",
        "Dense oBERT IMDB",
    ]
    multi_label_classification_models = [
        "Sparse oBERT GOEMOTIONS",
        "Dense oBERT GOEMOTIONS",
    ]

    # Tasks
    image_detection_tab = "Image Detection"
    image_classification_tab = "Image Classification"
    image_segmentation_tab = "Image Segmentation"
    sentiment_analysis_tab = "Sentiment Analysis"
    question_answering_tab = "Question Answering"
    token_classification_tab = "Token Classification"
    document_classification_tab = "Document Classification"
    multi_label_classification_tab = "Multi Label Text Classification"
    masked_language_modeling_tab = "Masked Language Modeling"

    # Labels
    model_label = "Select Model"
    engine_label = "Select Engine"
    batch_label = "Set Batch Size"
    time_label = "Set time (secs)"
    scenario_label = "Select Inference Scenario"
    button_label = "show me the 💵"
    output_label = "Benchmark Results"
    accordion_label = "This Machine Runs On..."

    # Parameters
    batch_min = 1
    batch_max = 128
    batch_step = 8
    batch_value = 1

    time_min = 1
    time_max = 60
    time_step = 2
    time_value = 5

    engines = ["DeepSparse", "ONNX"]
    scenarios = ["sync", "async"]

    # Map
    tab_switch = {
        image_detection_tab: image_detection_models,
        image_classification_tab: image_classification_models,
        image_segmentation_tab: image_segmentation_models,
        sentiment_analysis_tab: sentiment_analysis_models,
        question_answering_tab: question_answering_models,
        token_classification_tab: token_classification_models,
        document_classification_tab: document_classification_models,
        multi_label_classification_tab: multi_label_classification_models,
    }

    embd_video = """
    <html>
    <body>
    <div style="max-width: 700px;">
    <iframe
    width="100%" height="315"
    src="https://www.youtube.com/embed/gGErxSqf05o?autoplay=1&mute=1&modestbranding=0"
    title="YOLOv5 on CPUs: Sparsifying to Achieve GPU-Level Performance"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; web-share"
    allowfullscreen>
    </iframe>
    </div>
    </body>
    </html>
    """
