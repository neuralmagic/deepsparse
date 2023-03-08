import os
from deepsparse import Context, MultiModelEngine
from deepsparse.pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, SUPPORTED_PIPELINE_ENGINES, ORTEngine, Engine
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.helpers import overwrite_transformer_onnx_model_inputs

_MODEL_DIR_ONNX_DECODER_NAME = "decoder_model.onnx"


class AutoregressivePipeline(TransformersPipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        #assert os.path.exists(self.onnx_file_path[0]), f"Encoder onnx file does not exist at {self.onnx_file_path[0]}"
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
