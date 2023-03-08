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


class Manager:

    models = {
        "Sparse YOLOv5s COCO": "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none",  # noqa: E501
        "Dense YOLOv5s COCO": "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none",  # noqa: E501
        "Sparse YOLOv5m COCO": "zoo:cv/detection/yolov5-m/pytorch/ultralytics/coco/pruned70_quant-none",  # noqa: E501
        "Dense YOLOv5m COCO": "zoo:cv/detection/yolov5-m/pytorch/ultralytics/coco/base-none",  # noqa: E501
        "Sparse ResNet 50 ImageNet": "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_uniform_quant-none",  # noqa: E501
        "Dense ResNet 50 ImageNet": "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",  # noqa: E501
        "Sparse MobileNetV1 ImageNet": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate",  # noqa: E501
        "Dense MobileNetV1 ImageNet": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",  # noqa: E501
        "Sparse YOLACT COCO": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none",  # noqa: E501
        "Dense YOLACT COCO": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none",  # noqa: E501
        "Sparse DistilBERT SST2": "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned80_quant-none-vnni",  # noqa: E501
        "Dense DistilBERT SST2": "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/base-none",  # noqa: E501
        "Sparse oBERT SST2": "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none",  # noqa: E501
        "Dense oBERT SST2": "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none",  # noqa: E501
        "Sparse DistilBERT SQUAD": "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/pruned80_quant-none-vnni",  # noqa: E501
        "Dense DistilBERT SQUAD": "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/base-none",  # noqa: E501
        "Sparse oBERT SQUAD": "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none",  # noqa: E501
        "Dense oBERT SQUAD": "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/base-none",  # noqa: E501
        "Sparse DistilBERT CONLL": "zoo:nlp/token_classification/distilbert-none/pytorch/huggingface/conll2003/pruned80_quant-none-vnni",  # noqa: E501
        "Dense DistilBERT CONLL ": "zoo:nlp/token_classification/distilbert-none/pytorch/huggingface/conll2003/base-none",  # noqa: E501
        "Sparse oBERT CONLL": "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none",  # noqa: E501
        "Dense oBERT CONLL": "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none",  # noqa: E501
        "Sparse RoBERTa IMDB": "zoo:nlp/document_classification/roberta-base/pytorch/huggingface/imdb/pruned85_quant-none",  # noqa: E501
        "Dense RoBERTa IMDB": "zoo:nlp/document_classification/roberta-base/pytorch/huggingface/imdb/base_quant-none",  # noqa: E501
        "Sparse oBERT IMDB": "zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/pruned90_quant-none",  # noqa: E501
        "Dense oBERT IMDB": "zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/base-none",  # noqa: E501
        "Sparse oBERT GOEMOTIONS": "zoo:nlp/multilabel_text_classification/obert-base/pytorch/huggingface/goemotions/pruned90_quant-none",  # noqa: E501
        "Dense oBERT GOEMOTIONS": "zoo:nlp/multilabel_text_classification/obert-base/pytorch/huggingface/goemotions/base-none",  # noqa: E501
    }

    engines = {"DeepSparse": "deepsparse", "ONNX": "onnxruntime"}

    route = "/deepsparse"
