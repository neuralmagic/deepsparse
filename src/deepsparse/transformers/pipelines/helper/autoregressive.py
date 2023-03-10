import os
import numpy
from deepsparse import Context, MultiModelEngine
from deepsparse.pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, SUPPORTED_PIPELINE_ENGINES, ORTEngine, Engine
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.helpers import overwrite_transformer_onnx_model_inputs
from typing import Type, List, Mapping, Any
from pydantic import BaseModel
from abc import abstractmethod

_MODEL_DIR_ONNX_DECODER_NAME = "decoder_model.onnx"


class AutoregressivePipeline(TransformersPipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_engine()



    def engine_forward(self, engine_inputs: List[numpy.ndarray], **kwargs) -> List[numpy.ndarray]:
        """
        :param engine_inputs: list of numpy inputs to Pipeline engine forward
            pass
        :return: result of forward pass to Pipeline engine
        """
        batch_num = engine_inputs[0].shape[0]
        eos_token_found = [False] * batch_num
        new_tokens = []
        valid_tokens_mask = kwargs.get("valid_tokens_mask", None)
        if valid_tokens_mask is None:
            raise ValueError

        logits, *kv_cache = self.decoder_engine(engine_inputs)

        # Using the mask to keep the valid tokens only
        valid_tokens = numpy.ma.masked_array(engine_inputs[0], valid_tokens_mask)
        for batch_idx, valid_tokens_sequence in enumerate(valid_tokens):
            # by counting the number of valid tokens,
            # we can get the index of the last valid token
            # Is this assumption always valid?
            last_valid_token_idx = numpy.ma.count(valid_tokens_sequence)
            # get the logits that emerge after processing the last valid token
            last_logits = logits[batch_idx, last_valid_token_idx - 1, :]
            next_token = numpy.argmax(last_logits)
            eos_token_found[batch_idx] = next_token == self.tokenizer.eos_token_id
            if last_valid_token_idx >= self.sequence_length:
                raise ValueError("Sequence length exceeded")
            new_tokens.append(next_token)
            engine_inputs[1][batch_idx, last_valid_token_idx] = 1


        input_dict = {}
        input_dict['input_ids'] = numpy.array([[next_token]])
        input_dict['attention_mask'] = engine_inputs[1]

        kv_cache_names = [name.replace('present', 'past_key_values') for name in self.decoder_engine._output_names if name.startswith('present')]
        for name, array in zip(kv_cache_names, kv_cache):
            input_dict[name] = array

        engine_inputs = [input_dict[name] for name in self.onnx_input_names]

        while last_valid_token_idx < self.sequence_length:
            if all(eos_token_found):
                return engine_inputs[0]
            logits, *kv_cache = self.engine(engine_inputs)
            # Using the mask to keep the valid tokens only
            valid_tokens = numpy.ma.masked_array(engine_inputs[0], valid_tokens_mask)
            for batch_idx, valid_tokens_sequence in enumerate(valid_tokens):
                # by counting the number of valid tokens,
                # we can get the index of the last valid token
                # Is this assumption always valid?
                last_valid_token_idx = numpy.ma.count(valid_tokens_sequence)
                # get the logits that emerge after processing the last valid token
                last_logits = logits[batch_idx, last_valid_token_idx - 1, :]
                next_token = numpy.argmax(last_logits)
                eos_token_found = next_token == self.tokenizer.eos_token_id
                engine_inputs[0][batch_idx, last_valid_token_idx] = next_token
                engine_inputs[1][batch_idx, last_valid_token_idx] = 1

        return engine_inputs[0]



    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        onnx_path = super().setup_onnx_file_path()
        onnx_decoder_path = self.setup_decoder_onnx_file_path()
        return onnx_path, onnx_decoder_path


    def setup_decoder_onnx_file_path(self):
        decoder_onnx_path = os.path.join(self.model_path, _MODEL_DIR_ONNX_DECODER_NAME)

        decoder_onnx_path, self.decoder_onnx_input_names, self._temp_model_directory = overwrite_transformer_onnx_model_inputs(
            decoder_onnx_path, max_length=self.sequence_length
        )

        return decoder_onnx_path

    def _initialize_engine(self):
        assert len(self.onnx_file_path) == 2, "Expected two onnx files for encoder and decoder"
        assert os.path.exists(self.onnx_file_path[0]), f"Encoder onnx file does not exist at {self.onnx_file_path[0]}"
        assert os.path.exists(self.onnx_file_path[1]), f"Decoder onnx file does not exist at {self.onnx_file_path[1]}"
        self.engine = self._initialize_single_engine(onnx_file_path = self.onnx_file_path[0])
        self.decoder_engine = self._initialize_single_engine(onnx_file_path = self.onnx_file_path[1])


    def _initialize_single_engine(self, onnx_file_path):
        engine_type = self.engine_type.lower()

        if engine_type == DEEPSPARSE_ENGINE:
            if self.context is not None and isinstance(self.context, Context):
                self._engine_args.pop("num_cores", None)
                self._engine_args.pop("scheduler", None)
                self._engine_args["context"] = self.context
                return MultiModelEngine(
                    model=onnx_file_path,
                    **self._engine_args,
                )
            return Engine(onnx_file_path, **self._engine_args)
        elif engine_type == ORT_ENGINE:
            return ORTEngine(onnx_file_path, **self._engine_args)
        else:
            raise ValueError(
                f"Unknown engine_type {self.engine_type}. Supported values include: "
                f"{SUPPORTED_PIPELINE_ENGINES}"
            )

