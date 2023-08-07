import openai

# Modify OpenAI's API values to use the DeepSparse API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]

# Completion API
stream = True
completion = openai.Completion.create(
    model=model,
    prompt="def fib():",
    stream=stream,
    max_tokenss=32)

print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)
