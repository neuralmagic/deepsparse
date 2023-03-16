# DeepSparse Benchmark

This page explains how to use DeepSparse's CLI utilties for benchmarking performance in a variety of scenarios.

## Installation Requirements

Install DeepSparse with `pip`:

```bash
pip install deepsparse[onnxruntime]
```

The benchmarking numbers were achieved on an AWS `c6i.16xlarge` (32 core) instance.

## Quickstart

Let's compare DeepSparse's performance with dense and sparse models.

Run the following to benchmark DeepSparse with a [dense, unoptimized BERT ONNX model](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fbase-none):

```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none --batch_size 64

>> INFO:deepsparse.benchmark.benchmark_model:Starting 'singlestream' performance measurements for 10 seconds
>> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 102.1683
```

Run the following to benchmark DeepSparse with a [90% pruned and quantized BERT ONNX model](https://sparsezoo.neuralmagic.com/models/nlp%2Fsentiment_analysis%2Fobert-base%2Fpytorch%2Fhuggingface%2Fsst2%2Fpruned90_quant-none):

```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --batch_size 64

>> INFO:deepsparse.benchmark.benchmark_model:Starting 'singlestream' performance measurements for 10 seconds
>> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 889.1262
```

Running the sparse model, DeepSparse achieves 889 items/second vs 102 items/second with the dense model. **This is an 8.7x speedup!**

### Comparing to ONNX Runtime

The benchmarking utility also allows you to use ONNX Runtime as the inference runtime by passing `--engine onnxruntime`. 

Run the following to benchmark ORT with the same dense, unoptimized BERT ONNX model as above:
```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none --batch_size 64 --engine onnxruntime

>> INFO:deepsparse.benchmark.benchmark_model:Starting 'singlestream' performance measurements for 10 seconds
>> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 64.3392
```

Run the following to benchmark ORT with the same 90% pruned and quantized BERT ONNX model as above:
```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --batch_size 64 --engine onnxruntime

>> INFO:deepsparse.benchmark.benchmark_model:Starting 'singlestream' performance measurements for 10 seconds
>> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 55.1905
```

We can see that ORT does not gain additional performance from sparsity like DeepSparse. Additionally, DeepSparse runs the dense model
faster than ORT at high batch sizes. All in all, in this example **DeepSparse is 13.8x faster than ONNX Runtime**!

## Usage 

Run `deepsparse.benchmark -h` to see full command line arguments.

Let's walk through a few examples of common functionality.

### Pass Your Local ONNX Model

Beyond passing SparseZoo stubs, you can also pass a local path to an ONNX file to DeepSparse. As an example, let's download an ONNX file from SparseZoo using the CLI to a local directory called `./yolov5-download`.

```bash
sparsezoo.download zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none --save-dir yolov5-download
```

We can pass a local ONNX file as follows:
```bash
deepsparse.benchmark yolov5-download/model.onnx
>> Original Model Path: yolov5-download/model.onnx
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 219.7396
```

### Batch Sizes

We can adjust the batch size of the inference with `-b` or `--batch_size`.

The following runs a 95% pruned-quantized version of ResNet-50 at batch size 1:
```bash
deepsparse.benchmark zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none --batch_size 1

>> Original Model Path: zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 852.7742
```

The following runs a 95% pruned-quantized version of ResNet-50 at batch size 64:
```bash
deepsparse.benchmark zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none --batch_size 64

>> Original Model Path: zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none
>> Batch Size: 64
>> Scenario: sync
>> Throughput (items/sec): 2456.9958
```

In general, DeepSparse is able to achieve better performance at higher batch sizes, especially on many core machines as it is better able to utilitize the underlying hardware and saturate all of the cores at high batch sizes.

### Custom Input Shape

We can adjust the input share of the inference with `-i` or `--input_shape`. This is generally useful for changing the size of input images or sequence length for NLP.

Here's an example doing a BERT inference with sequence length 384 (vs 128 as above):

```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --input_shape [1,384]

>> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 121.7578
```

Here's an example doing a YOLOv5s inference with a 320x320 image (rather than 640x640 as above)

```bash
deepsparse.benchmark zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none -i [1,3,320,320]

>> Original Model Path: zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 615.7185
```

### Inference Scenarios

The default scenrio is synchronous inference. Set by the `--scenario sync` argument, the goal metric is latency per batch (ms/batch). This scenario submits a single inference request at a time to the engine, recording the time taken for a request to return an output. This mimics an edge deployment scenario.

Additionally, DeepSparse offers asynchronous inference, where DeepSparse will allocate resources to handle multiple inferences at once. Set by the `--scenario async` argument. This scenario submits `--num_streams` concurrent inference requests to the engine. This mimics a model server deployment scenario.

Here's an example handling 8 concurrent batch 1 inferences:

```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --scenario async --num_streams 8

>> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
>> Batch Size: 1
>> Scenario: async
>> Throughput (items/sec): 807.3410
>> Latency Mean (ms/batch): 9.8906
```

Here's an example handling one batch 1 inference at a time with the same model:

```bash
deepsparse.benchmark zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none --scenario sync

>> Original Model Path: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 269.6001
>> Latency Mean (ms/batch): 3.7041
```

We can see that the async scenario achieves higher throughput, while the synchronous scenario achieves lower latency. Especially for very high core counts, using the asynchronous scheduler is a great way to improve performance if running at low batch sizes.
