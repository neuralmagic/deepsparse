import pytest
from deepsparse.loggers.metric_functions import sequence_length

@pytest.mark.parametrize(
    "sequence, expected_len",
    [
        ("His palms are sweaty", 20),
        (["knees weak","arms are heavy"], {"0": 10, "1": 14}),
    ]
)
def test_sequence_length(sequence, expected_len):
    assert sequence_length(sequence) == expected_len