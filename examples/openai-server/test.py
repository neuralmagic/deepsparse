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

# Make sure to start the server first:
"""
python examples/openai-server/server.py \
    --model zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none \
    --prompt-processing-sequence-length 1
"""  # noqa: E501

import unittest

import openai


class TestOpenAIApi(unittest.TestCase):
    def setUp(self):
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"

    def test_list_models(self):
        # Fetch available model
        models = openai.Model.list()
        self.assertIsNotNone(models)
        self.assertIn("data", models)
        return [model["id"] for model in models["data"]]

    def test_model_completion(self):
        models = self.test_list_models()

        for model in models:
            response = openai.Completion.create(
                model=model, prompt="def fib():", max_tokens=16
            )
            self.assertIsNotNone(response)
            self.assertIn("choices", response)
            self.assertTrue(len(response["choices"]) > 0)
            print(response["choices"])

    def test_streaming_output(self):
        models = self.test_list_models()

        for model in models:
            responses = openai.Completion.create(
                model=model, prompt="def fib():", max_tokens=16, stream=True
            )
            for response in responses:
                self.assertIn("choices", response)
                self.assertTrue(len(response["choices"]) > 0)
                print(response["choices"])


if __name__ == "__main__":
    unittest.main()
