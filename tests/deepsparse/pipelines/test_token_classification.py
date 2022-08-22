from deepsparse import Pipeline
import pytest


@pytest.fixture(scope="module")
def pipeline():
    yield Pipeline.create("token_classification")


_EXPECTED_OUTPUT = [
    [
        "my",
        "name",
        "is",
        ("bob", "LABEL_1"),
        "and",
        "i",
        "live",
        "in",
        ("boston", "LABEL_5"),
    ],
    [
        ("sally", "LABEL_1"),
        ("bob", "LABEL_2"),
        ("joe", "LABEL_2"),
        ".",
        ("california", "LABEL_5"),
        ",",
        ("united", "LABEL_5"),
        ("states", "LABEL_6"),
        ".",
        ("europe", "LABEL_5"),
    ],
]


def test_token_classification_pipeline(pipeline):
    output = pipeline(
        inputs=[
            "My name is Bob and I live in Boston",
            "Sally Bob Joe. California, United States. Europe",
        ]
    )
    assert len(output.predictions) == len(_EXPECTED_OUTPUT)
    for sentence_words, expected_words in zip(output.predictions, _EXPECTED_OUTPUT):
        assert len(sentence_words) == len(expected_words)
        for result, expected in zip(sentence_words, expected_words):
            if isinstance(expected, tuple):
                expected_word, expected_label = expected
            else:
                expected_word, expected_label = expected, "LABEL_0"
            assert result.word == expected_word, expected_words
            assert result.entity == expected_label, result
