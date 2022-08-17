import pytest
from PIL import Image
from deepsparse.pipelines import CVSchema
import numpy
from tests.deepsparse.pipelines.data_helpers import computer_vision


def _get_images():
    batch_size = 5
    images = computer_vision(batch_size=batch_size)
    return images.get("images")


def test_accepts_input():
    CVSchema(images="asdf")
    CVSchema(images=["asdf", "qwer"])
    CVSchema(images=numpy.zeros((1, 2, 3)))
    CVSchema(images=[numpy.zeros((1, 2, 3)), numpy.zeros((1, 2, 3))])


@pytest.mark.parametrize(
    "schema_cls, image_files", [(CVSchema, _get_images())]
)
def test_from_files(schema_cls, image_files):
    image_iters = (open(image, "rb") for image in image_files)

    expected = schema_cls(
        images=[numpy.array(Image.open(image)) for image in image_files]
    )
    actual = schema_cls.from_files(files=image_iters)

    assert isinstance(actual, schema_cls)
    assert len(actual.images) == len(expected.images)

    for actual_img, expected_img in zip(actual.images, expected.images):
        assert actual_img.shape == expected_img.shape
