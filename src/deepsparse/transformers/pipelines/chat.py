from deepsparse.operators import OperatorRegistry
from deepsparse.transformers.pipelines.text_generation import TextGenerationPipeline
from contextlib import contextmanager

__all__ = ["ChatPipeline"]

@OperatorRegistry.register(name=["chat", "chatbot"])
class ChatPipeline(TextGenerationPipeline):
    def __init__(self, **kwargs):
        self.storage_kv_cache = SessionStorageKVCache()
        super().__init__(**kwargs)


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
    


