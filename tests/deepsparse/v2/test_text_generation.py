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
from deepsparse.v2.text_generation import TextGenerationPipeline
from deepsparse.transformers.utils.helpers import prepends_bos_token


@pytest.fixture
def pipeline():
    input_arguments = dict(
        model_path="hf:mgoin/TinyStories-1M-deepsparse",
        engine_type="onnxruntime",
    )
    return TextGenerationPipeline(**input_arguments)


@pytest.fixture
def prompt():
    return "Never gonna give you up, never gonna let you down"


def test_freeze_first_position(pipeline):
    # Test whether we should be "freezing" the first token after
    # the kv cache is full
    assert not prepends_bos_token(pipeline.tokenizer)


# Damian: why can't we just pass output_scores=True to the pipeline
# call
# Damian: why can't we just pass pipeline(prompt, **kwargs)
def test_run_same_prompt_multiple_times(pipeline, prompt):
    # Test the scenario, where the same prompt is run multiple times
    # Every run should produce the same output
    output_1 = pipeline(prompt = prompt, generation_kwargs = dict(output_scores=True))
    output_2 = pipeline(prompt = prompt, generation_kwargs = dict(output_scores=True))

    assert output_1.generations[0].text == output_2.generations[0].text
    assert numpy.allclose(
        output_1.generations[0].score,
        output_2.generations[0].score,
        atol=1e-3,
    )

@pytest.mark.skip(reason="Running multiple prompts in "
                         "parallel returns one generation")
def test_run_multiple_prompts_in_parallel(pipeline, prompt):
    # Test the scenario, where multiple prompts are run in parallel
    # Same two prompts should produce the same output

    output = pipeline(prompt = [prompt, prompt], generation_kwargs = dict(output_scores=True))

    logits_0 = output.generations[0].score
    sequence_0 = output.generations[0].text

    logits_1 = output.generations[1].score
    sequence_1 = output.generations[1].text

    assert numpy.allclose(logits_0, logits_1, atol=1e-3)
    assert sequence_0 == sequence_1

@pytest.mark.skip(reason="num generated predictions does not work yet")
def test_num_generated_predictions(pipeline, prompt):
    # Test the scenario, where multiple predictions are generated
    # from the same prompt

    output_sequences = pipeline(prompt = prompt, num_return_sequences=2)

    assert len(output_sequences.generations) == 1
    assert len(output_sequences.generations[0]) == 2

    output_sequences = pipeline(prompt = [prompt, prompt], num_return_sequences=2)
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
    inference = pipeline(prompt = prompt, num_return_sequences=3, do_sample=True)
    generations = inference.generations
    # Output should be different from one another
    text_outputs = [x.text for x in generations[0]]
    assert len(set(text_outputs)) == 3

@pytest.mark.skip(reason="Streaming not yet implemented")
def test_streaming_mode_returns_generator(pipeline, prompt):
    response_generator = pipeline(prompt = prompt, streaming=True)
    assert inspect.isgenerator(
        response_generator
    ), "Pipeline should return a generator in streaming mode"

    assert all(
        isinstance(response, pipeline.output_schema) for response in response_generator
    ), "Pipeline should return a generator of output_schema \
           objects in streaming mode"