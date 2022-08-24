from pydantic import BaseModel, Field
from typing import Optional


class DeepSparseSchema(BaseModel):
    inference_timing: Optional["InferenceTimingSchema"] = Field(
        description="blabla")