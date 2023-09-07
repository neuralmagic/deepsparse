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

from typing import List

import numpy

from deepsparse.utils.data import numpy_softmax


class TokenGenerator:
    def __init__(
        self,
        logits: numpy.ndarray,
        tokens: List[int] = [],
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs,
    ):
        self.token_frequencies = numpy.zeros(logits.shape[-1])

        self.deterministic = deterministic
        self.sampling_termperature = sampling_temperature
        self.top_k = top_k
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.tokens = []
        for token in tokens:
            self.update_frequencies(token)

    def update_frequencies(self, token: numpy.ndarray):
        self.tokens.append(token)
        self.token_frequencies[token] += 1

    def generate(self, logits: numpy.ndarray) -> numpy.ndarray:
        """
        Samples a token from the logits using the sampling temperature.

        :param logits: the logits from the model with shape (vocab_size,)
        :return: the sampled token
        """
        if self.deterministic:
            token = numpy.argmax(logits)
            self.tokens.append(token)
            return token

        if self.sampling_temperature != 1.0:
            logits /= self.sampling_temperature

        if self.top_k:
            logits = self.apply_top_k(logits)
        if self.top_p:
            logits = self.apply_top_p(logits)

        # penalties here
        if self.frequency_penalty != 0.0:
            logits = self.apply_frequency_penalty(logits)
        if self.presence_penalty != 0.0:
            logits = self.apply_presence_penalty(logits)

        probs = self.numpy_softmax(logits)
        token = numpy.random.choice(len(probs), p=probs)
        self.update_frequencies(token)

        return token

    def apply_frequency_penalty(self, logits: numpy.ndarray):
        logits -= self.frequency_penalty * self.token_frequencies
        return logits

    def apply_presence_penalty(self, logits: numpy.ndarray):
        logits -= self.presence_penalty * (self.token_frequencies > 0)
        return logits

    # from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    def apply_top_k(self, logits: numpy.ndarray, filter_value=-float("Inf")):
        logits_shape = logits.shape
        logits = logits.reshape(logits.shape[-1])
        top_k_indices = numpy.argpartition(logits, -self.top_k)[-self.top_k :]
        logits[~numpy.isin(numpy.arange(len(logits)), top_k_indices)] = filter_value

        return logits.reshape(logits_shape)

    # from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    def apply_top_p(
        self, logits: numpy.ndarray, filter_value=-float("Inf"), min_tokens_to_keep=1
    ):
        logits_shape = logits.shape
        logits = logits.reshape(logits.shape[-1])

        sorted_indices = numpy.argsort(logits)
        sorted_logits = logits[sorted_indices]
        logit_cumulative_probs = numpy.cumsum(numpy_softmax(sorted_logits))

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = logit_cumulative_probs > self.top_p
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

        return logits.reshape(logits_shape)
