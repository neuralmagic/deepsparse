from deepsparse.transformers.pipelines.question_answering import (
    QuestionAnsweringPipeline,
    QuestionAnsweringOutput,
    QuestionAnsweringInput,
)
from deepsparse import Pipeline

TASK = "custom_qa_task"


@Pipeline.register(TASK)
class MyCustomQaPipeline(QuestionAnsweringPipeline):
    def parse_inputs(self, *args, **kwargs) -> QuestionAnsweringInput:
        qa_input: QuestionAnsweringInput = super().parse_inputs(*args, **kwargs)
        # do whatever you want here
        qa_input.question = qa_input.question.replace("whom", "who")
        return qa_input

    def process_engine_outputs(self, *args, **kwargs) -> QuestionAnsweringOutput:
        qa_output: QuestionAnsweringOutput = super().process_engine_outputs(
            *args, **kwargs
        )
        # do whatever you want here
        qa_output.answer = qa_output.answer.replace("bob", "joe")
        return qa_output
