from deepsparse import Pipeline
import pytest


@pytest.fixture(scope="module")
def pipeline():
    yield Pipeline.create("question_answering")


def test_question_answering_pipeline(pipeline):
    output = pipeline(question="who am i", context="i am corey")
    assert output.answer == "corey"
