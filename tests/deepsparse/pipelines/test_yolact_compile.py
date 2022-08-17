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
    dynamic_pipeline = Pipeline.create(
        "yolact", model_path=stub, batch_size=None, executor=ThreadPoolExecutor()
    )
    static_pipeline = Pipeline.create(task="yolact", batch_size=2)
    assert dynamic_pipeline.use_dynamic_batch()
    assert not static_pipeline.use_dynamic_batch()
