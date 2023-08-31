# test perplexity script
# test what's going on after we go past sequence_length
# sequences need to be appropriately long to notice the divergence over time
# also test kv cache storage (continuous inference)

# config 
# nightly 


# metric to evaluate logits and cache -> min absolute difference
# maybe if env var set, we can also plot graphs

OUR_MODELS = ["opt", "codegen", "llama"]
for model in OUR_MODELS:

    # establish sources of truth
    torch_target_logits = ...
    ort_target_logits = ...
    torch_target_cache = ...
    ort_target_cache = ...

    for engine_type in ["onnxruntime", "deepsparse"] 
        ort_no_kv_cache_logits = ...
        deepsparse_no_kv_cache_logits = ... # no-kv cache models in stubs.

        if kv_cache:
            for kv_cache_management in ["external", "internal"]

            DONE ort_single_token_prefill_logits = ...
            DONE ort_multi_token_prefill_logits = ...
            DONE ort_single_token_prefill_cache = ...
            DONE ort_multi_token_prefill_cache = ...

            DONE ds_single_token_prefill_logits_external = ...
            DONE ds_multi_token_prefill_logits_external = ...
            DONE ds_single_token_prefill_cache_external = ...
            DONE ds_multi_token_prefill_cache_external = ...

            DONE ds_single_token_prefill_logits_internal = ...
            DONE ds_multi_token_prefill_logits_internal = ...



            
            

        


    
