# Pipeline Creation

```python
from deepsparse import TextGeneration

MODEL_PATH = "path/to/model/or/zoostub"
text_pipeline = TextGeneration(model_path=MODEL_PATH)

```

# Inference Runs

```python
PROMPT = "how are you?"
SECOND_PROMPT = "what book is really popular right now?"
```

### All defaults 
```python
text_result = text_pipeline(prompt=PROMPT)
```

### Enable Streaming
```python
genrations = text_pipeline(prompt=PROMPT, streaming=True)
for text_generation in genrations:
    print(text_generation)
```

### Multiple Inputs
```python
PROMPTS = [PROMPT, SECOND_PROMPT]
generations = text_pipeline(prompt=PROMPTS)
prompt_output = generations[0]
second_prompt_output = generation[1]
```

### Use the GenerationConfig
#### Limit the generated output size using the `max_length` property

```python

generation_config = {"max_length": 10}
generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
print(generations)

```
### Use just kwargs

```python

generations = text_pipeline(prompt=PROMPT, max_length=10)
print(generations)

```

### Get more then one response

```python
generation_config = {"num_return_sequences": 2}
generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
print(generations)
```

### Get more than one unique response

```python
generation_config = {"num_return_sequences": 2, "do_sample": True}
generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
print(generations)
```

### Output scores

```python
generations = text_pipeline(prompt=PROMPT, output_score=True)
print(generations)
```

<h1><summary>Text Generation GenerationConfig Features Supported </h1></summary>

<details>
<h2> Parameters controlling the output length: </h2>

| Feature | Description | Deepsparse Default | HuggingFace Default | Supported |
| :---    |      :----: |         :----:     |        :----:       |       ---:|
| max_length | Maximum length of generated tokens. Equal to input_prompt + max_new_tokens. Overridden by max_new_tokens | 1024 | 20 | Yes|
| max_new_tokens | Maximum number of tokens to generate, ignoring prompt tokens. | None | None | Yes |
| min_length | Minimum length of generated tokens. Equal to input_prompt + min_new_tokens. Overridden by min_new_tokens | - | 0 | No
| min_new_tokens | Minomum number of tokens to generate, ignoring prompt tokens. | - | None | No |
| max_time | - | - | - | No |

<br/>
<h2> Parameters for manipulation of the model output logits </h2>

| Feature | Description | Deepsparse Default | HuggingFace Default | Supported |
| :---    |      :----: |         :----:     |        :----:       |       ---:|
| top_k | The number of highest probability vocabulary tokens to keep for top-k-filtering | 0 | 50 | Yes
| top_p | Keep the generated tokens where its cumulative probability is >= top_p | 0.0 | 1.0 | Yes
| repetition_penalty | Penalty applied for generating new token. Existing token frequencies summed to subtraction the logit of its corresponding logit value | 0.0 | 1.0 | Yes |
| temperature | The temperature to use when sampling from the probability distribution computed from the logits. Higher values will result in more random samples. Should be greater than 0.0 | 1.0 | 1.0 | Yes |
| typical_p | - | - | - | No |
| epsilon_cutoff | - | - | - | No |
| eta_cutoff | - | - | - | No |
| diversity_penalty | - | - | - | No |
| length_penalty | - | - | - | No |
| bad_words_ids | - | - | - | No |
| force_words_ids | - | - | - | No |
| renormalize_logits | - | - | - | No |
| constraints | - | - | - | No |
| forced_bos_token_id | - | - | - | No |
| forced_eos_token_id | - | - | - | No |
| remove_invalid_values | - | - | - | No |
| exponential_decay_length_penalty | - | - | - | No |
| suppress_tokens | - | - | - | No |
| begin_suppress_tokens | - | - | - | No |
| forced_decoder_ids | - | - | - | No |

<br/>
<h2> Parameters that control the generation strategy used </h2>

| Feature | Description | Deepsparse Default | HuggingFace Default | Supported |
| :---    |      :----: |         :----:     |        :----:       |       ---:|
| do_sample | If True, will apply sampling from the probability distribution computed from the logits | False | False | Yes |

<br/>
<h2> Parameters for output variables: </h2>

| Feature | Description | Deepsparse Default | HuggingFace Default | Supported |
| :---    |      :----: |         :----:     |        :----:       |       ---:|
| num_return_sequences | The number of sequences generated for each prompt | 1 | 1 | Yes |
| output_scores | Whether to return the generated logits | False | False | Yes |
| return_dict_generate | - | - | - | No |

<br/>
<h2> Special Tokens: </h2>

| Feature | Description | Deepsparse Default | HuggingFace Default | Supported |
| :---    |      :----: |         :----:     |        :----:       |       ---:|
| pad_token_id | - | - | - | No |
| bos_token_id | - | - | - | No |
| eos_token_id | - | - | - | No |

</details>
<br/>
