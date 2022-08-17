from deepsparse import Pipeline


def test_compile_yolact():
    stub = (
        "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none"
    )
    Pipeline.create("yolact", model_path=stub)
