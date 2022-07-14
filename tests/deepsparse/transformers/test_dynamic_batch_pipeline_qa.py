from concurrent.futures import ThreadPoolExecutor
import pytest
from deepsparse import Pipeline


@pytest.mark.parametrize("task", ["text_classification"], scope="class")
class TestDynamicBatchPipeline:
    def test_dynamic_batch_threaded_pipeline_creation(self, task):
        Pipeline.create(
            task=task,
            batch_size=None,
            threadpool=ThreadPoolExecutor(),
        )

    def test_dynamic_batch_pipeline_creation_without_threadpool_raises_value_error(
        self,
        task,
    ):
        with pytest.raises(ValueError) as _value_error:
            Pipeline.create(
                task=task,
                batch_size=None,
            )

    @pytest.mark.parametrize("batch_size", [10])
    def test_order_retention(self, task, batch_size):
        # Create test samples

        inputs = Pipeline.create(
            task=task,
        ).input_schema.create_sample_inputs(batch_size)

        # Run each sample through its own pipeline
        static_outputs = []
        static_batch_threaded_pipeline = Pipeline.create(
            task=task,
            batch_size=1,
            threadpool=ThreadPoolExecutor(),
        )

        for _input in inputs:
            static_outputs.append(static_batch_threaded_pipeline(_input).result())

        # Run batch through dynamic pipeline
        dynamic_batch_threaded_pipeline = Pipeline.create(
            task=task,
            batch_size=None,
            threadpool=ThreadPoolExecutor,
        )
        dynamic_outputs = dynamic_batch_threaded_pipeline(inputs).result()

        # Check that order is maintained
        assert static_outputs == dynamic_outputs
