from src.deepsparse.evaluation.utils import Evaluation, Dataset, Metric, EvalSample
import pytest
import numpy as np


@pytest.mark.parametrize(
    "list_of_evaluations",
    [Evaluation(task="task_1",
                 dataset=Dataset(type="type_1",
                                 name="name_1",
                                 config="config_1",
                                 split ="split_1"
                                 ),
                 metrics = "metric_1",
                 eval_sample= EvalSample(input=np.array([[5]]),
                                         output = 5
                 )),
      Evaluation(task="task_2",
                dataset=Dataset(type="type_2",
                                name="name_2",
                                config="config_2",
                                split ="split_2"
                                ),
                 metrics = ["metric_1", "metric_2"],
                 eval_sample= [EvalSample(input=np.array([[10.]]),
                                          output = 10.),
                               EvalSample(input=np.array([[20.]]),
                                           output = 20.)])])


def test_serialize_evaluation_json(list_of_evaluations):
    [evaluation.dict() for evaluation in list_of_evaluations]



