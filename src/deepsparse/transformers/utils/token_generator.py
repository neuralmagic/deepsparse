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

from typing import List, Optional

import numpy

from deepsparse.utils.data import numpy_softmax


_MIN_FLOAT = numpy.finfo(numpy.float32).min


class TokenGenerator:
    """
    Responsible for generating tokens, and contains functions that
    token generation depends on including different sampling and
    filtering methods
    """

    def __init__(
        self,
        logits_shape: int,
        tokens: Optional[List[int]] = None,
        deterministic: bool = True,
        sampling_temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs,
    ):
        """
        :param logits_shape: int representing the size/length of the logit
         to be used. Note that generated token will have the upper bound of
         this value
        :param tokens: Any previously generated tokens. Used to keep frequncy counts
         to be used for penalty calculations
        :param deterministic: set to  True will always return the same output with the
         same inputs
        :param sampling_temperature: used to add randomness to the generated token
        :param top_k: select top_k logit values
        :param top_p: select the cumulative sum of the logits values outside of top_p
        :param frequency_penalty: subtract its value and its token frequency count
         to thelogit
        :param presence_penalty: subtract any corresponding logit with existing tokens
        """
        self.token_frequencies = numpy.zeros(logits_shape)

        self.deterministic = deterministic
        self.sampling_temperature = sampling_temperature
        self.top_k = top_k
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.tokens = [] if tokens is None else tokens

        self._initialize_token_frequencies()

    def generate(self, logits: numpy.ndarray) -> numpy.ndarray:
        """
        Samples a token from the logits. If non-deterministic, logits that tokens
        get generated from will be a function of sampling_temperature, top_k, top_p,
        frequency_penalty and presence_penalty.

        :param logits: the logits from the model with shape (vocab_size,)
        :return: the sampled token
        """

        if self.deterministic:
            token = numpy.argmax(logits)
            self.tokens.append(token)
            return token

        # make a copy of logits to avoid modifying the original
        # logits distribution in-place
        logits = logits.copy()

        if self.top_k:
            logits = self.apply_top_k(logits)

        if self.top_p:
            logits = self.apply_top_p(logits)

        if self.sampling_temperature != 1.0:
            logits /= self.sampling_temperature

        if self.frequency_penalty != 0.0:
            logits = self.apply_frequency_penalty(logits)
        if self.presence_penalty != 0.0:
            logits = self.apply_presence_penalty(logits)

        probs = numpy_softmax(logits)
        token = numpy.random.choice(len(probs), p=probs)

        self.tokens.append(token)
        self._update_frequencies(token)

        return token

    def apply_frequency_penalty(self, logits: numpy.ndarray) -> numpy.ndarray:
        """Apply frequency_penalty based on the token frequency count"""
        logits -= self.frequency_penalty * self.token_frequencies
        return logits

    def apply_presence_penalty(self, logits: numpy.ndarray) -> numpy.ndarray:
        """
        Apply prensence_penaly to any logits where there exists
        a token
        """
        logits -= self.presence_penalty * (self.token_frequencies > 0)
        return logits

    # from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf31
    def apply_top_k(
        self, logits: numpy.ndarray, filter_value=_MIN_FLOAT
    ) -> numpy.ndarray:
        """
        Keep top_k logits based on its value. All other values
        will be overwritten to filter_value

        :param filter_value: value to overwrite non-top_k values
        """
        logits_shape = logits.shape
        logits = logits.reshape(logits.shape[-1])
        top_k_indices = numpy.argpartition(logits, -self.top_k)[-self.top_k :]
        logits[~numpy.isin(numpy.arange(len(logits)), top_k_indices)] = filter_value

        return logits.reshape(logits_shape)

    # from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    def apply_top_p(
        self,
        logits: numpy.ndarray,
        filter_value=_MIN_FLOAT,
        min_tokens_to_keep: int = 1,
    ) -> numpy.ndarray:
        """
        Keep any logits' cumulative sum <= top_p. non top_p logits will be
        overwritten to filter_value

        :param filter_value: value to overwrite non-top_p values
        :param min_tokens_to_keep: number of logit values to keep to avoid
         zero valued logits
        """
        logits_shape = logits.shape
        logits = logits.reshape(logits.shape[-1])

        sorted_indices = numpy.argsort(logits)
        sorted_logits = logits[sorted_indices]
        logit_cumulative_probs = numpy.cumsum(numpy_softmax(sorted_logits))

        # Remove tokens with cumulative top_p above the threshold
        # (token with 0 are kept)
        sorted_indices_to_remove = logit_cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        if min_tokens_to_keep:
            sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        return logits.reshape(logits_shape)

    def _update_frequencies(self, token: numpy.ndarray):
        self.token_frequencies[token] += 1

    def _initialize_token_frequencies(self):
        unique_tokens, frequencies = numpy.unique(self.tokens, return_counts=True)
        for token, freq in zip(unique_tokens, frequencies):
            self.token_frequencies[token] += freq
