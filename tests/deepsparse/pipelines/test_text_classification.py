from deepsparse import Pipeline
import pytest


@pytest.fixture(scope="module")
def pipeline():
    yield Pipeline.create("text_classification")


def test_text_classification_pipeline(pipeline):
    output = pipeline(
        sequences=["this is a good sentence", "this sentence is terrible!"]
    )
    assert output.labels == ["LABEL_1", "LABEL_0"]
    assert len(output.scores) == 2
