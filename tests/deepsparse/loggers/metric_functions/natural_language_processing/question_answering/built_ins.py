import pytest
from deepsparse.transformers.pipelines.question_answering import QuestionAnsweringOutput
from deepsparse.loggers.metric_functions import answer_length, answer_score

output_schema = QuestionAnsweringOutput(answer = "His palms are sweaty", score = 0.69, start = 0, end = 0)
@pytest.mark.parametrize(
    "schema, expected_len",
    [
        (output_schema, 20),
    ]
)
def test_answer_length(schema, expected_len):
    assert answer_length(schema) == expected_len

@pytest.mark.parametrize(
    "schema, expected_score",
    [
        (output_schema, 0.69),
    ]
)
def test_answer_score(schema, expected_score):
    assert answer_score(schema) == expected_score