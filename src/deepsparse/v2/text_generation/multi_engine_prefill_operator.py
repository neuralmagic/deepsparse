from deepsparse.v2.operators import Operator


class MultiEnginePrefill(Operator):
    def __init__(self, prompt_sequence_length):
        self.prompt_sequence_length = prompt_sequence_length
        self.sequence_length = sequence_length
        self.onnx_input_names_no_cache = onnx_input_names_no_cache
    
    def run(self, inp: Any, context: Optional[Context]):
        ## process token_batch
        ## return to run through multi_engie; store prompt logits + tokens processed (pipeline variable?)
        ## who is looping through the token batches and keeping track?


