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
"""
pip install transformers[onnx]
optimum-cli export onnx --model bigscience/bloom-560m --task causal-lm-with-past bloom-560m/
cd bloom-560m/
mv decoder_with_past_model.onnx model.onnx
"""
import os
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, pipeline


def preprocess_shape(s):
    past_sequence_length = 0
    batch_size = 1
    # for some reason we have 16 heads,
    # not sure how to get this number
    num_heads = 16
    if s == "past_sequence_length":
        return past_sequence_length
    elif s == "batch_size x num_heads":
        return batch_size * num_heads
    elif s == "past_sequence_length + 1":
        return past_sequence_length + 1
    elif s == "batch_size":
        return batch_size
    assert isinstance(s, int)
    return s


def autoregressive_pass(current_token, context_len, model, kv_cache, kv_output_names):
    inputs = defaultdict(np.array)
    inputs["input_ids"] = np.array([[current_token]])
    inputs["attention_mask"] = np.ones((1, context_len), dtype=np.int64)

    logits, *new_kvs = model.run(["logits"] + kv_output_names, {**inputs, **kv_cache})
    kv_cache = dict(zip(kv_output_names, new_kvs))
    kv_cache = {k.replace("present", "past_key_values"): v for k, v in kv_cache.items()}
    return logits, kv_cache


def main(
    input_sequence: str = "Why is Batman always",
    max_new_tokens=50,
    model_name="bigscience/bloom-560m",
    model_path="bloom-560m",
):

    generator = pipeline("text-generation", model=model_name)
    ground_truth = generator(input_sequence, max_new_tokens=max_new_tokens)

    #### Recreate the pipeline in ONNX Runtime ####

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    x = tokenizer(input_sequence, return_tensors="np").data
    tokens = x["input_ids"].tolist()[0]
    output_tokens = tokens

    model = ort.InferenceSession(os.path.join(model_path, "model.onnx"))

    kv_output_names = [n.name for n in model.get_outputs() if "present" in n.name]

    kv_cache = defaultdict(np.ndarray)
    for input_ in model.get_inputs():
        name = input_.name
        if name.startswith("past_key_values"):
            shape_list = input_.shape
            shape = [preprocess_shape(s) for s in shape_list]
            kv_cache[name] = np.zeros(shape, dtype=np.float32)

    for i, token in enumerate(tokens):
        context_len = i + 1
        logits, kv_cache = autoregressive_pass(
            token, context_len, model, kv_cache, kv_output_names
        )

    current_token = np.argmax(logits[0, -1, :])
    output_tokens.append(current_token)

    for j in range(max_new_tokens - 1):
        if current_token == tokenizer.eos_token_id:
            break
        context_len = i + j + 1 + 1

        logits, kv_cache = autoregressive_pass(
            current_token, context_len, model, kv_cache, kv_output_names
        )

        current_token = np.argmax(logits[0, -1, :])
        output_tokens.append(current_token)

    print("#" * 50)
    prediction = "".join(
        tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    )
    print(prediction)
    print(ground_truth)
    print(
        "ORT prediction matches ground truth pipeline: ",
        prediction == ground_truth[0]["generated_text"],
    )
    print("#" * 50)


if __name__ == "__main__":
    main()
