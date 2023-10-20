from deepsparse.v2.operators import Operator
from deepsparse.transformers.utils.helpers import create_causal_mask
from pydantic import BaseModel
import numpy

__all__ = ["AutoRegressiveOperator"]

class AutoRegressiveInput(BaseModel):
    tokens: Any = Field(description="tokens")
    kv_cache: DecoderKVCache = Field(description="kv_cache object")

class AutoRegressiveOutput(BaseModel):
    engine_inputs: list = Field(description="engine inputs maps")

class AutoRegressiveOperator(Operator):
    input_schema = AutoRegressiveInput
    output_schema = AutoRegressiveOutput

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def run(inp: Any, context: Optional[Context]):
        kv_cache = inp.kv_cache
        tokens = inp.tokens
        engine_input_names = inp.engine_input_names ## property of the engine

        num_total_processed_tokens = kv_cache.total_num_processed_tokens
        new_token = tokens[-1]
        # padding is added to left, so attention mask is 1s from the
        # right up to the number of total tokens (prompt + generated)
        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        num_attention_entries_to_unmask = min(
            num_total_processed_tokens + 1, self.sequence_length
        )  # cap by seq len
        attention_mask[:, -num_attention_entries_to_unmask:] = 1
        positions = numpy.array([[num_total_processed_tokens]], dtype=numpy.int64)
        input_ids = numpy.array([[new_token]])
        causal_mask = create_causal_mask(input_ids, attention_mask)

        # filter out the inputs that are not needed by the engine
        engine_inputs_map = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            positions=positions,
        )
        
        engine_inputs = [
            engine_inputs_map[name] for name in engine_input_names
        ]

        return {"engine_inputs": engine_inputs}


        """ next operator to call engine
        generated_logits = self.engine(engine_inputs, kv_cache)

        return generated_logits
        """
    