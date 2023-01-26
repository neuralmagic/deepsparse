import pytest
from tests.utils import mock_engine
from deepsparse import Pipeline

yaml_config = """
loggers:
    list_logger:
        path: tests/deepsparse/loggers/helpers.py:ListLogger
data_logging:
    predefined:
    - func: question_answering
      frequency: 1"""

expected_logs = """identifier:question_answering/pipeline_inputs.question__string_length, value:10, category:MetricCategories.DATA
identifier:question_answering/pipeline_inputs.context__string_length, value:18, category:MetricCategories.DATA
identifier:question_answering/pipeline_outputs__answer_found, value:True, category:MetricCategories.DATA
identifier:question_answering/pipeline_outputs__answer_length, value:13, category:MetricCategories.DATA
identifier:question_answering/pipeline_outputs__answer_score, value:1.8941500186920166, category:MetricCategories.DATA"""  # noqa E501

@pytest.mark.parametrize(
    "config, inp, num_iterations, expected_logs",
    [

        (yaml_config, ("Go, shorty", "It's your birthday"), 1, expected_logs),
    ],
)
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config, inp, num_iterations, expected_logs):
    pipeline = Pipeline.create("qa", logger=config)
    for _ in range(num_iterations):
        question, context = inp
        pipeline(question=question, context=context)

    logs = pipeline.logger.loggers[0].logger.loggers[0].calls
    data_logging_logs = [log for log in logs if "DATA" in log]
    for log, expected_log in zip(data_logging_logs, expected_logs.splitlines()):
        assert log == expected_log