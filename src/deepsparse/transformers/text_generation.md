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
generations = text_pipeline(prompt=PROMPT, streaming=True)
for text_generation in generations:
    print(text_generation)
```

### Multiple Inputs
```python
PROMPTS = [PROMPT, SECOND_PROMPT]
text_output = text_pipeline(prompt=PROMPTS)

prompt_output = text_output.generations[0]
second_prompt_output = text_output.generations[1]
```

### Use `generation_config` to control the generated results
- Limit the generated output size using the `max_length` property
- For a complete list of supported attributes, see the tables below

```python
generation_config = {"max_length": 10}
generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
```

### Use the transformers `GenerationConfig` object for the `generation_config`

```python
from transformers import GenerationConfig

generation_config = GenerationConfig()
generation_config.max_length = 10

generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
```

### Use just `kwargs`
- The attributes supported through the `generation_config` are also supported through
`kwargs`

```python
generations = text_pipeline(prompt=PROMPT, max_length=10)
```
### Use the GenerationConfig during pipeline creation
- Every inference run with this pipeline will apply this generation config, unless
also provided during inference

```python
MODEL_PATH = "path/to/model/or/zoostub"
generation_config = {"max_length": 10}
text_pipeline = TextGeneration(model_path=MODEL_PATH, generation_config=generation_config)

generations = text_pipeline(prompt=PROMPT)

# Override the generation config by providing a config during inference time
generation_config = {"max_length": 25}
generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
```

### Get more then one response for a given prompt

```python
generation_config = {"num_return_sequences": 2}
generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
```

### Get more than one unique response

```python
generation_config = {"num_return_sequences": 2, "do_sample": True}
generations = text_pipeline(prompt=PROMPT, generation_config=generation_config)
```

### Use multiple prompts and generate multiple outputs for each prompt

```python
PROMPTS = [PROMPT, SECOND_PROMPT]

generations = text_pipeline(prompt=PROMPTS, num_return_sequences=2, do_sample=True, max_length=100)
prompt_outputs = text_output.generations[0]
second_prompt_outputs = text_output.generations[1]

print("Outputs from the first prompt: ")
for output in prompt_outputs:
   print(output)
   print("\n")

print("Outputs from the second prompt: ")
for output in second_prompt_outputs:
   print(output)
   print("\n")
```

Output:
```
Outputs from the first prompt:
text="  are you coping better with holidays?\nI'm been reall getting good friends and helping friends as much as i can so it's all good." score=None finished=True finished_reason='stop'

text="\nI'm good... minor panic attacks but aside from that I'm good." score=None finished=True finished_reason='stop'

Outputs from the second prompt: 
text='\nHAVING A GOOD TIME by Maya Angelou; How to Be a Winner by Peter Enns; BE CAREFUL WHAT YOU WHORE FOR by Sarah Bergman; 18: The Basic Ingredients of a Good Life by Jack Canfield.\nI think you might also read The Sympathy of the earth by Charles Darwin, if you are not interested in reading books. Do you write? I think it will help you to refine your own writing.' score=None finished=True finished_reason='stop'

text='  every school or publication I have looked at has said the same two books.\nIt depends on the school/master. AIS was the New York Times Bestseller forever, kicked an ass in the teen fiction genre for many reasons, a lot of fiction picks like that have been around a while hence popularity. And most science fiction and fantasy titles (but not romance or thriller) are still popular.' score=None finished=True finished_reason='stop'
```


### Output scores

```python
generations = text_pipeline(prompt=PROMPT, output_score=True)
```

<h1><summary>Text Generation GenerationConfig Features Supported </h1></summary>


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


<br/>
