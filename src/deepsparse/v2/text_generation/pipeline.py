from deepsparse.v2 import pipeline

class TextGenerationPipeline(Pipeline):
    def __init__(self, prompt_sequence_length, enable_multitoken_prefill):
        transformers_preprocess = TransformersPreprocess() ## set-up config/tokenizer
        
        single_engine_operator = NLEngineOperator() # set-up engine
        multi_engine_operator = NLEngineOperator(enable_multitoken_prefill) # set-up engine

        input_preprocess = Preprocess() # take in config/produce input_tokens, update context variable
        tokens_to_engine_input = TokenToEngineInput() ## convert input_tokens to engine_input (depends on if multi-token available)

        
        prefill_preprocess = PrefillPreprocess() 
        ## schema with engine specific values? or the engine itself?
        # class variables?
        # get tokens based on attn mask, set-up kv_cache, return both
        
        ## Update the schema as part of the run function in the pipeline?
        ## that would put the ownness on the pipeline to be able to check which engine can run/is valid
        



        ops = {"input_preprocess": input_preprocess, 
            "prefill_preprovess": prefill_preprocess,
            "singe_engine_operator": single_engine_operator, 
            "multi_engine_operator": multi_engine_operator}
        
        routes = {"preprocess": {"input_preprocess", "token_to_engine_input"}}





            
            


        

        



      

