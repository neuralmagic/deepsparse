    
class Manager:
    models = {
        "Sparse YOLOv5s COCO": "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none",
        "Dense YOLOv5s COCO": "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none",
        "Sparse YOLOv5m COCO": "zoo:cv/detection/yolov5-m/pytorch/ultralytics/coco/pruned70_quant-none",
        "Dense YOLOv5m COCO": "zoo:cv/detection/yolov5-m/pytorch/ultralytics/coco/base-none",
        "Sparse ResNet 50 ImageNet": "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_uniform_quant-none",
        "Dense ResNet 50 ImageNet": "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",
        "Sparse MobileNetV1 ImageNet": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate",
        "Dense MobileNetV1 ImageNet": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",
        "Sparse YOLACT COCO": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none",
        "Dense YOLACT COCO": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none",
        "Sparse DistilBERT SST2": "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned80_quant-none-vnni",
        "Dense DistilBERT SST2": "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/base-none",
        "Sparse oBERT SST2": "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none",
        "Dense oBERT SST2": "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none",
        "Sparse DistilBERT SQUAD": "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/pruned80_quant-none-vnni",
        "Dense DistilBERT SQUAD": "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/base-none",
        "Sparse oBERT SQUAD": "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none",
        "Dense oBERT SQUAD": "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/base-none",
        "Sparse DistilBERT CONLL": "zoo:nlp/token_classification/distilbert-none/pytorch/huggingface/conll2003/pruned80_quant-none-vnni",
        "Dense DistilBERT CONLL ": "zoo:nlp/token_classification/distilbert-none/pytorch/huggingface/conll2003/base-none",
        "Sparse oBERT CONLL": "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/pruned90_quant-none",
        "Dense oBERT CONLL": "zoo:nlp/token_classification/obert-base/pytorch/huggingface/conll2003/base-none",
        "Sparse RoBERTa IMDB": "zoo:nlp/document_classification/roberta-base/pytorch/huggingface/imdb/pruned85_quant-none",
        "Dense RoBERTa IMDB": "zoo:nlp/document_classification/roberta-base/pytorch/huggingface/imdb/base_quant-none",
        "Sparse oBERT IMDB": "zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/pruned90_quant-none",
        "Dense oBERT IMDB": "zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/base-none",
        "Sparse oBERT GOEMOTIONS": "zoo:nlp/multilabel_text_classification/obert-base/pytorch/huggingface/goemotions/pruned90_quant-none",
        "Dense oBERT GOEMOTIONS": "zoo:nlp/multilabel_text_classification/obert-base/pytorch/huggingface/goemotions/base-none",
    }
    
    engines = {"DeepSparse": "deepsparse", "ONNX": "onnxruntime"}
    
    route = "/deepsparse"
