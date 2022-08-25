from deepsparse import Pipeline, BucketingPipeline
import pytest
from tests.utils import mock_engine


@mock_engine(rng_seed=0)
def test_question_answering_choose_bucket(engine_mock):
    pipeline20 = Pipeline.create("question_answering", sequence_length=20)
    pipeline50 = Pipeline.create("question_answering", sequence_length=50)

    pipeline = BucketingPipeline([pipeline20, pipeline50])

    # this should have token length == 14 and should be routed to pipeline20
    bucket, _ = pipeline._choose_bucket(question="a " * 5, context="b " * 5)
    assert bucket is pipeline20

    # should should have token length 44, and should be routed to pipeline50
    bucket, _ = pipeline._choose_bucket(question="a " * 20, context="b " * 20)
    assert bucket is pipeline50


@pytest.mark.parametrize(
    "task",
    [
        "text_classification",
        "token_classification",
        "embedding_extraction",
        "zero_shot_text_classification",
    ],
)
@mock_engine(rng_seed=0)
def test_text_choose_bucket(engine_mock, task):
    pipeline20 = Pipeline.create(task, sequence_length=20)
    pipeline50 = Pipeline.create(task, sequence_length=50)

    pipeline = BucketingPipeline([pipeline20, pipeline50])

    bucket, _ = pipeline._choose_bucket("a " * 10)
    assert bucket is pipeline20

    bucket, _ = pipeline._choose_bucket("a " * 35)
    assert bucket is pipeline50

    bucket, _ = pipeline._choose_bucket(["a " * 10, "a " * 35])
    assert bucket is pipeline50

    bucket, _ = pipeline._choose_bucket(["a " * 10, "a " * 12])
    assert bucket is pipeline20
