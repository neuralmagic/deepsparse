from mlserver import MLModel
from mlserver.codecs import decode_args
from typing import List
from deepsparse import Pipeline

class DeepSparseRuntime(MLModel):
    async def load(self) -> bool:
        self._pipeline = Pipeline.create(
            task = self._settings.parameters.task,
            model_path = self._settings.parameters.model_path,
            batch_size = self._settings.parameters.batch_size,
            sequence_length = self._settings.parameters.sequence_length,
        )
        return True
    
    @decode_args
    async def predict(self, sequences: List[str]) -> List[str]:
        prediction = self._pipeline(sequences=sequences)
        return prediction.labels