from deepsparse import Pipeline
from concurrent.futures import ThreadPoolExecutor


def test_compile_yolact():
    stub = (
        "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none"
    )
    Pipeline.create("yolact", model_path=stub)


def test_compile_yolact_dynamic_batch():
    stub = (
        "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none"
    )
    Pipeline.create(
        "yolact", model_path=stub, batch_size=None, executor=ThreadPoolExecutor()
    )
