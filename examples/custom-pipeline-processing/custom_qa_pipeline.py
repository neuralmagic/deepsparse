# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from deepsparse import Pipeline
from deepsparse.transformers.pipelines.question_answering import (
    QuestionAnsweringInput,
    QuestionAnsweringOutput,
    QuestionAnsweringPipeline,
)


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
