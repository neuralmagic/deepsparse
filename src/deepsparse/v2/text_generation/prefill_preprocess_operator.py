from pydantic import BaseModel
from deepsparse.v2.operators import operator

class PrefillPreprocessInput(BaseModel):
    engine_inputs: list = Field(description="engine inputs")

class PrefillPreprocessOutput(BaseModel):
    tokens: Any = Field(description="tokens")
    kv_cache: DecoderKVCache = Field(description="kv_cache object")

class PrefillPreprocess(Operator):
    input_schema = None
    output_schema = None

    def run(self, inp: Any, context: Optional[Context]):
        engine_inputs = inp.engine_inputs

        tokens = engine_inputs[0][engine_inputs[1].nonzero()].tolist()
        kv_cache = get_kv_cache_decoder(...) # requires engine attributes, engine as input?

        return {"tokens": tokens, "kv_cache": kv_cache}
