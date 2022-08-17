from deepsparse import Pipeline
from concurrent.futures import ThreadPoolExecutor


def test_compile_yolact():
    Pipeline.create("yolact")


def test_compile_yolact_dynamic_batch():
    dynamic_pipeline = Pipeline.create(
        "yolact", batch_size=None, executor=ThreadPoolExecutor()
    )
    static_pipeline = Pipeline.create(task="yolact", batch_size=2)
    assert dynamic_pipeline.use_dynamic_batch()
    assert not static_pipeline.use_dynamic_batch()
