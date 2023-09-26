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

import contextvars
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy
from pydantic import Field, validator

from deepsparse import Pipeline
from deepsparse.transformers.pipelines.text_generation import (
    FinishReason,
    TextGenerationInput,
    TextGenerationOutput,
    TextGenerationPipeline,
)
from deepsparse.transformers.utils import (
    DecoderKVCache,
    SessionStorageKVCache,
    create_causal_mask,
    generate_session_id,
    prepends_bos_token,
    validate_session_ids,
)
from deepsparse.utils.data import split_engine_inputs


_LOGGER = logging.getLogger(__name__)
_SESSION_IDS_CONTEXT = contextvars.ContextVar("_SESSION_ID", default=None)

__all__ = ["ChatPipeline"]


class ChatInput(TextGenerationInput):
    session_ids: Union[None, List[str], str] = Field(
        default=None,
        description="String identifier(s) "
        "for the kv cache session(s). If None, "
        "and the model is using kv cache, session_id "
        "will be set to a random uuid.",
    )

    @validator("session_ids")
    def validate_session_ids(cls, value, values) -> Union[None, List[str]]:
        # make sure that the that format session_ids confirms with the
        # rest of the inputs
        session_ids = validate_session_ids(session_ids=value, other_attributes=values)
        return session_ids


class ChatOutput(TextGenerationOutput):
    session_ids: Union[None, str, List[str]] = Field(
        default=None, description="A string identifier(s) for the kv cache session."
    )


@Pipeline.register(
    task="chat",
    task_aliases=["chatbot"],
)
class ChatPipeline(TextGenerationPipeline):
    """
    Pipeline for chat applications using transformers models.

    The chat pipeline keeps a persistent kv cache sessions that
    can be used to store information between inference calls.
    This to enable the chatbot to "recollect" information
    from previous conversations. The kv cache sessions are
    stored in the SessionStorageKVCache.

    This pipeline is a subclass of the TextGenerationPipeline
    and adds the following functionality:

    - on `process_engine_outputs(..)` it adds the session ids
      to the engine inputs
    - on `split_engine_inputs(..)` it distributes the session id
        to the appropriate batch
    - on `engine_forward(..)` it enables to fetch the existing session
        from the StorageKVCache (if the appropriate session for the
        session id exists, else create new session). It also adds the
        session id to the engine outputs.
    - on `join_engine_outputs(..)` it adds the session ids to the
        aggregated engine outputs
    - on `process_engine_outputs(..)` it adds the session ids to the
        output schema
    """

    def __init__(self, **kwargs):
        self.storage_kv_cache = SessionStorageKVCache()
        super().__init__(**kwargs)

    @property
    def input_schema(self) -> Type[ChatInput]:
        """
        Property to return the input schema for the pipeline.

        :return: the input schema for the pipeline
        """
        return ChatInput

    @property
    def output_schema(self) -> Type[ChatOutput]:
        """
        Property to return the output schema for the pipeline.

        :return: the output schema for the pipeline
        """
        return ChatOutput

    @contextmanager
    def session(
        self,
        session_ids: Union[None, List[str], str] = None,
        inference_batch_size: int = 1,
    ) -> Callable[[Any, Any], Any]:
        """
        Context manager that sets and keeps a default session id(s) within
        the context

        example:
        In the following - both responses in the context will share the same
        session id
        ```
        with chat_pipeline.session():
            first_response = chat_pipeline("first prompt")
            second_response = chat_pipeline("second prompt")
        ```

        :param session_ids: actual value to set session ids to in context
            must match the inference batch size. If not supplied, will
            create default values. Default None
        :param inference_batch_size: if generating default session ids, number
            of session ids to create. default 1
        """

        if session_ids is None:
            session_ids = [generate_session_id() for _ in range(inference_batch_size)]

        # set session_ids contextvar
        token = _SESSION_IDS_CONTEXT.set(session_ids)
        yield
        # reset session_ids contextvar
        _SESSION_IDS_CONTEXT.reset(token)

    def process_inputs(
        self, inputs: ChatInput
    ) -> Tuple[List[numpy.ndarray], Dict[str, Any]]:
        """
        Add session ids information to the original engine input

        :param inputs: the input schema for the pipeline
        :return: the inputs for the engine
        """
        engine_input, context = super().process_inputs(inputs)
        return self.add_session_ids_to_engine_input(engine_input, inputs), context

    def process_engine_outputs(
        self, engine_outputs: List[Union[numpy.ndarray, FinishReason, str]], **kwargs
    ) -> ChatOutput:
        """
        Convert the engine outputs to the output schema for the pipeline.

        :param engine_outputs: the outputs from the engine
        :param kwargs: additional keyword arguments
        :return: the output schema for the pipeline
        """

        engine_outputs, session_ids = list(*engine_outputs)

        kwargs["session_ids"] = session_ids
        # process the engine outputs within the context of TextGenerationPipeline
        return super().process_engine_outputs(engine_outputs, **kwargs)

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], context: Dict
    ) -> Tuple[numpy.ndarray, numpy.ndarray, List[FinishReason], str]:
        """
        Wrapper around the engine forward function to add session ids to
        the returned engine outputs

        :param engine_inputs: the inputs for the engine
        :param context: the context for the engine
        :return: the outputs from the engine
        """
        session_id = engine_inputs[-1]
        if session_id in self.storage_kv_cache and prepends_bos_token(self.tokenizer):
            # if the session exists and the model prepends bos token
            # we need to remove the bos token from the engine inputs
            # to make sure that kv cache session is continuous
            # between inference calls
            engine_inputs = self.undo_bos_token(engine_inputs)

        return *super().engine_forward(engine_inputs, context), session_id

    def get_kv_cache_decoder(self, engine_inputs: List[Any]) -> DecoderKVCache:
        """
        Initialize the kv cache decoder for the inference.
        If the storage kv cache does not have a kv cache decoder,
        create a new one and store it in the storage kv cache.

        :param engine_inputs: the inputs for the engine
        :return: the initialized kv cache decoder
        """
        session_id = engine_inputs[-1]
        kv_cache = self.storage_kv_cache.get(session_id)
        if kv_cache is None:
            # use the super class to initialize the kv cache decoder
            kv_cache: DecoderKVCache = super().get_kv_cache_decoder(engine_inputs)
            self.storage_kv_cache[session_id] = kv_cache

        return kv_cache

    def undo_bos_token(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        Undo the prepending of the bos token to the input sequence
        by the TextGenerationPipeline.

        :param engine_inputs: the inputs for the engine
        :return: the inputs for the engine without the bos token
        """
        engine = self.engine or self.multitoken_engine

        engine_inputs_no_bos = []

        for idx, name in enumerate(engine.onnx_input_names_no_cache):
            if name == "input_ids" or "attention_mask":
                # remove first (bos) token/entry from input_ids
                engine_input = engine_inputs[idx][:, 1:]
            elif name == "positions":
                # remove the last position
                # e.g. positions = [[0,1,2,3]]
                # ->   positions = [[0,1,2]]
                engine_input = engine_inputs[idx][:, :-1]
            elif name == "causal_mask":
                # recompute the causal mask
                input_ids = engine_inputs_no_bos[0]
                attention_mask = engine_inputs_no_bos[1]
                engine_input = create_causal_mask(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            engine_inputs_no_bos.append(engine_input)
        # finally add the session id
        engine_inputs_no_bos.append(engine_inputs[-1])
        return engine_inputs_no_bos

    def add_session_ids_to_engine_input(
        self, engine_input: List[numpy.ndarray], inputs: ChatInput
    ):
        """
        Create the session ids for the input sequences (if not provided)
        and add them to the engine input.

        :param engine_input: the inputs for the engine
        :param inputs: the input schema for the pipeline

        :return: the engine input with the session ids
        """
        session_ids = inputs.session_ids
        if session_ids is None and _SESSION_IDS_CONTEXT.get() is not None:
            # respect directly setting session IDs first, then try to pull
            # from context
            session_ids = _SESSION_IDS_CONTEXT.get()
        elif session_ids is None:
            # session_ids is None, so we need to generate
            # a session id for each input sequence
            # TODO: Talk to Dipika whether this aligns with the
            #       feature where we create multiple outputs for
            #       a single input sequence
            num_input_sequences = (
                len(inputs.sequences) if isinstance(inputs.sequences, list) else 1
            )
            session_ids = [generate_session_id() for _ in range(num_input_sequences)]
        engine_input.append(session_ids)
        return engine_input

    def join_engine_outputs(
        self,
        batch_outputs: List[List[Union[numpy.ndarray, FinishReason, str]]],
        orig_batch_size: int,
        **context,
    ) -> List[Union[numpy.ndarray, FinishReason, str]]:
        """
        Wrapper around the join_engine_outputs function that handles
        the session_ids that are returned from the engine.

        :param batch_outputs: the outputs from the engine
        :param orig_batch_size: the original batch size
        :return: the joined outputs from the engine
        """
        # get the session_ids from the batch_outputs
        *batch_outputs, session_ids = list(zip(*batch_outputs))
        # unzip batch_outputs and feed them back to super class
        batch_outputs = list(zip(*batch_outputs))
        batch_outputs = super().join_engine_outputs(
            batch_outputs, orig_batch_size, **context
        )
        # add the session_ids to the batch_outputs
        yield batch_outputs, session_ids

    def split_engine_inputs(
        self, items: List[Union[numpy.ndarray, List[str]]], batch_size: int
    ) -> Tuple[List[List[numpy.ndarray]], int]:
        """
        Custom implementation of splitting the engine inputs that takes into
        account the fact that the `items` contain additionally a list of
        session_ids, that need to be distributed across the batches.
        :param items: list of numpy arrays to split (plus list of session_ids)
        :param batch_size: size of each batch to split into
        :return: list of batches, where each batch is a list of numpy arrays
            (plus session_ids), as well as the total batch size
        """
        # extract the session_ids from the items
        session_ids = next((item for item in items if isinstance(item, list)), None)
        items = [item for item in items if not isinstance(item, list)]

        batches, orig_batch_size = split_engine_inputs(items, batch_size)

        # distribute session_ids across batches
        batches_w_session_ids = [
            batch + [session_ids[i]] for i, batch in enumerate(batches)
        ]

        return batches_w_session_ids, orig_batch_size
