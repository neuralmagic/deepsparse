import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import pytest
from deepsparse import Pipeline


@pytest.fixture()
def threadpool():
    yield ThreadPoolExecutor()


@pytest.mark.parametrize("task", ["text_classification"], scope="class")
class TestDynamicBatchPipeline:
    @pytest.fixture()
    def dynamic_batch_pipeline(self, task, threadpool):
        yield Pipeline.create(
            task=task,
            batch_size=None,
            threadpool=threadpool,
        )

    def static_batch_pipeline(self, task, threadpool, batch_size=1):
        return Pipeline.create(
            task=task,
            batch_size=batch_size,
            threadpool=threadpool,
        )

    def test_dynamic_batch_threaded_pipeline_creation(self, dynamic_batch_pipeline):
        # Will fail if fixture request fails
        pass

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
    def test_dynamic_batch_pipeline_returns_a_future(
        self,
        dynamic_batch_pipeline,
        batch_size,
    ):
        inputs = dynamic_batch_pipeline.input_schema.create_sample_inputs(
            batch_size=batch_size,
        )
        output = dynamic_batch_pipeline(**inputs)
        assert isinstance(output, concurrent.futures.Future), (
            "Expected dynamic batch  pipeline to return a concurrent.futures.Future"
            f" object but got {type(output)} instead"
        )

    @pytest.mark.parametrize("batch_size", [10])
    def test_order_retention(self, task, threadpool, batch_size, dynamic_batch_pipeline):
        inputs = Pipeline.create(
            task=task,
        ).input_schema.create_sample_inputs(batch_size)

        # Run each sample through its own pipeline
        static_batch_threaded_pipeline = self.static_batch_pipeline(
            task=task,
            batch_size=batch_size,
            threadpool=threadpool,
        )
        static_outputs = static_batch_threaded_pipeline(**inputs).result()
        dynamic_outputs = dynamic_batch_pipeline(**inputs).result()

        # Check that order is maintained
        assert static_outputs == dynamic_outputs
