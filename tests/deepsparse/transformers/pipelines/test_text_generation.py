# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import numpy

import pytest

# TODO: update to use/be compliant with new pipeline
from deepsparse.legacy.pipeline import Pipeline
from deepsparse.transformers.utils.helpers import prepends_bos_token


@pytest.fixture
def pipeline():
    return Pipeline.create(
        task="text_generation",
        model_path="hf:mgoin/TinyStories-1M-deepsparse",
        engine_kwargs={"engine_type": "onnxruntime"},
    )


@pytest.fixture
def prompt():
    return "Never gonna give you up, never gonna let you down"


def test_freeze_first_position(pipeline):
    # Test whether we should be "freezing" the first token after
    # the kv cache is full
    assert not prepends_bos_token(pipeline.tokenizer)


def test_run_same_prompt_multiple_times(pipeline, prompt):
    # Test the scenario, where the same prompt is run multiple times
    # Every run should produce the same output
    output_1 = pipeline(prompt, output_scores=True)
    output_2 = pipeline(prompt, output_scores=True)

    assert output_1.generations[0].text == output_2.generations[0].text
    assert numpy.allclose(
        output_1.generations[0].score,
        output_2.generations[0].score,
        atol=1e-3,
    )


def _test_stop_inference_kv_cache_full(
    pipeline,
    prompt,
    max_new_tokens,
    expected_finished_reason,
    expected_generated_tokens_length=None,
):
    out = pipeline(prompt=prompt, max_new_tokens=max_new_tokens)
    kv_cache_state = out.kv_cache_state[0]
    finished_reason = out.generations[0].finished_reason
    generated_text = out.generations[0].text
    assert finished_reason == expected_finished_reason
    assert len(pipeline.tokenizer(generated_text)["input_ids"]) == (
        expected_generated_tokens_length or max_new_tokens
    )
    return kv_cache_state


def test_stop_inference_kv_cache_full(prompt):
    # Tests the proper behavior of the kv cache around the
    # scenario when the kv cache becomes full during the inference

    # We set the sequence length to a small value to assert that
    # the kv cache buffer fills up quickly
    sequence_length = 32
    # We set the prompt sequence length to 1 to assert that the
    # inference will run until the kv cache is full. If the
    # `prompt_sequence_length` is larger than 1, it is very probable
    # that the inference will stop before the kv cache is full
    # (as the `prompt_sequence_length` reduces the number of
    # tokens that are generated in the first iteration)
    prompt_sequence_length = 1

    pipeline = Pipeline.create(
        task="text_generation",
        model_path="hf:mgoin/TinyStories-1M-deepsparse",
        engine_type="onnxruntime",
        sequence_length=sequence_length,
        force_max_tokens=True,
        prompt_sequence_length=prompt_sequence_length,
    )
    pipeline._debug = True

    prompt_length = len(pipeline.tokenizer(prompt)["input_ids"])

    cache_capacity = sequence_length - prompt_sequence_length
    # we need to subtract 1 to account for the initial generated token during the
    # prompt inference
    cache_capacity -= 1

    # max_new_tokens so that there is still one more "free" space in the kv cache
    # (we can still do autoregressive inference)
    max_new_tokens_minus_one = cache_capacity - prompt_length - 1
    # max_new_tokens so that the kv cache is full
    # (so we can still do one last correct autoregressive
    # inference in the next iteration)
    max_new_tokens = cache_capacity - prompt_length
    # max_new_tokens so that kv cache has already removed the last entry
    # (so we can no longer do autoregressive inference in the next iteration)
    max_new_tokens_plus_one = cache_capacity - prompt_length + 1
    # max_new_tokens so that kv cache would remove two last entries
    # (but it will not, the inference terminates early and produces
    # the same result as max_new_tokens_plus_one)
    max_new_tokens_plus_two = cache_capacity - prompt_length + 2

    kv_cache_state_full_minus_one = _test_stop_inference_kv_cache_full(
        pipeline,
        prompt,
        max_new_tokens_minus_one,
        expected_finished_reason="max_new_tokens",
    )
    kv_cache_state_full = _test_stop_inference_kv_cache_full(
        pipeline, prompt, max_new_tokens, expected_finished_reason="max_new_tokens"
    )
    kv_cache_state_full_plus_one = _test_stop_inference_kv_cache_full(
        pipeline, prompt, max_new_tokens_plus_one, expected_finished_reason="capacity"
    )
    kv_cache_state_full_plus_two = _test_stop_inference_kv_cache_full(
        pipeline,
        prompt,
        max_new_tokens_plus_two,
        expected_generated_tokens_length=max_new_tokens_plus_one,
        expected_finished_reason="capacity",
    )
    """
    Check the following structure ok the kv cache:
    minus_one | full | plus_one | plus_two
    --------------------------------------
     [- 0 -] | [row A] | [row B] | [row B]
     [row A] | [row B] | [row C] | [row C]
     [row B] | [row C] | [row D] | [row D]
       ...   |   ...   |   ...   |  ...
    """
    # check for the "free" space in the kv cache
    assert kv_cache_state_full_minus_one["past_key_values.0.key"][:, :, 0, :].sum() == 0
    # check for the row A
    assert numpy.array_equal(
        kv_cache_state_full_minus_one["past_key_values.0.key"][:, :, 1, :],
        kv_cache_state_full["past_key_values.0.key"][:, :, 0, :],
    )
    # check for the row B
    assert numpy.array_equal(
        kv_cache_state_full["past_key_values.0.key"][:, :, 1, :],
        kv_cache_state_full_plus_one["past_key_values.0.key"][:, :, 0, :],
    )
    # check equality between plus_one and plus_two
    assert numpy.array_equal(
        kv_cache_state_full_plus_one["past_key_values.0.key"],
        kv_cache_state_full_plus_two["past_key_values.0.key"],
    )


def test_run_multiple_prompts_in_parallel(pipeline, prompt):
    # Test the scenario, where multiple prompts are run in parallel
    # Same two prompts should produce the same output

    output = pipeline([prompt, prompt], output_scores=True)

    logits_0 = output.generations[0].score
    sequence_0 = output.generations[0].text

    logits_1 = output.generations[1].score
    sequence_1 = output.generations[1].text

    assert numpy.allclose(logits_0, logits_1, atol=1e-3)
    assert sequence_0 == sequence_1


def test_num_generated_predictions(pipeline, prompt):
    # Test the scenario, where multiple predictions are generated
    # from the same prompt

    output_sequences = pipeline(prompt, num_return_sequences=2)

    assert len(output_sequences.generations) == 1
    assert len(output_sequences.generations[0]) == 2

    output_sequences = pipeline([prompt, prompt], num_return_sequences=2)
    assert len(output_sequences.generations) == 2

    for generation in output_sequences.generations:
        assert len(generation) == 2


def test_token_generation_deterministic(pipeline, prompt):
    inference = pipeline(prompt, num_return_sequences=3, do_sample=False)
    generations = inference.generations
    # Output should be the same from one another
    text_outputs = [x.text for x in generations[0]]
    assert len(set(text_outputs)) == 1


def test_token_generation_non_deterministic(pipeline, prompt):

    inference = pipeline(prompt, num_return_sequences=3, do_sample=True)
    generations = inference.generations
    # Output should be different from one another
    text_outputs = [x.text for x in generations[0]]
    assert len(set(text_outputs)) == 3


def test_pipeline_for_ppl_eval(pipeline, prompt):
    predictions = pipeline(
        prompt,
        output_scores=True,
        return_input_tokens=True,
        fixed_sequences_length=True,
        include_prompt_logits=True,
        max_length=1,
    )
    assert hasattr(predictions, "generations")
    assert hasattr(predictions, "input_tokens")
    assert hasattr(predictions.generations[0], "score")
    assert "input_ids" in predictions.input_tokens
    assert "attention_mask" in predictions.input_tokens


def test_streaming_mode_returns_generator(pipeline, prompt):
    response_generator = pipeline(prompt, streaming=True)
    assert inspect.isgenerator(
        response_generator
    ), "Pipeline should return a generator in streaming mode"

    assert all(
        isinstance(response, pipeline.output_schema) for response in response_generator
    ), "Pipeline should return a generator of output_schema \
           objects in streaming mode"


def test_streaming_with_several_prompts(pipeline, prompt):
    additional_prompt = "Never gonna run around and desert you"
    prompts = [prompt, additional_prompt]

    generations_first_prompt_only = list(pipeline(prompt=prompts[0], streaming=True))
    generations_second_prompt_only = list(pipeline(prompt=prompts[1], streaming=True))

    bag_of_words_first_prompt = [
        g.generations[0].text for g in generations_first_prompt_only
    ]
    bag_of_words_second_prompt = [
        g.generations[0].text for g in generations_second_prompt_only
    ]

    generations = pipeline(prompt=prompts, streaming=True)
    bag_of_words_shared = []
    for r in generations:
        for gen in r.generations:
            text = gen.text
            bag_of_words_shared.append(text)

    assert sorted(bag_of_words_first_prompt + bag_of_words_second_prompt) == sorted(
        bag_of_words_shared
    )
