from typing import Type, List, Union, Tuple, Dict, Any

import numpy
from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.transformers.pipelines.helper import AutoregressivePipeline



__all__ = ["TextGenerationPipeline"]

@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen"],
)
class TextGenerationPipeline(AutoregressivePipeline):

    @staticmethod
    def route_input_to_bucket(*args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs) -> Pipeline:
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_inputs(self, inputs: BaseModel) -> Union[
        List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        pass

    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray], **kwargs) -> BaseModel:
        pass

    @property
    def input_schema(self) -> Type[BaseModel]:
        pass

    @property
    def output_schema(self) -> Type[BaseModel]:
        pass


