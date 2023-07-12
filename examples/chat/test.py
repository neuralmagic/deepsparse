from examples.codegen.text_generation import TextGenerationPipeline
# from transformers import TextStreamer


ort_codegen = TextGenerationPipeline(
    model_path="codegen-350M-multi",
    engine_type="onnxruntime",
    sequence_length=1024,
    num_tokens_to_generate=128,
    streamer=True)

ds_codegen = TextGenerationPipeline(
    model_path="codegen-350M-multi",
    engine_type="deepsparse",
    sequence_length=1024,
    num_tokens_to_generate=128,
    streamer=True)

print(ds_codegen.engine)


import time

start = time.perf_counter()
out = ort_codegen(sequences="def hello_world():")
end = time.perf_counter()
print(f"ORT took {end-start} seconds")
print(out.sequences[0])

start = time.perf_counter()
out = ds_codegen(sequences="def hello_world():")
end = time.perf_counter()
print(f"DeepSparse took {end-start} seconds")
print(out.sequences[0])


"""
def hello_world():
    print('Hello World!')
    print('Hello World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!


def hello_world():
    print('Hello World!')
    print('Hello World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
    print('Hello World!')
    print('Hello World!')
    print('Hello World!')
    print('World!')
    print('Hello World!')
"""
