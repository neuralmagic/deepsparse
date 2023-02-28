from deepsparse.transformers import QuestionAnsweringPipeline
import datasets
import numpy
from transformers.data import SquadExample
from transformers.models.auto import AutoConfig, AutoTokenizer
from transformers.data.processors.squad import (
    squad_convert_example_to_features,
    squad_convert_example_to_features_init,
)
from tqdm import tqdm
import time
from cuad_eval import CUAD

PATH = "/nm/drive3/tuan/models/CUAD/sparse_transfer/potential/CUAD@oberta-pruned90@sparse_transfer_decay_cuad@EP10@BS32@H1.0@T5.0@WD0.0001@LR8e-5@ID20943/checkpoint-1536/deployment"


class CUADPipeline(QuestionAnsweringPipeline):
    def setup_onnx_file_path(self):
        path = super().setup_onnx_file_path()
        self.slow_tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", use_fast=False, add_prefix_space=True
        )
        # self.slow_tokenizer.add_prefix_space = True

        # old_tokenize = self.slow_tokenizer.tokenize

        # def new_tokenize(*args, **kwargs):
        #     kwargs.pop("add_prefix_space", None)
        #     return old_tokenize(*args, **kwargs)

        # self.slow_tokenizer.tokenize = new_tokenize
        squad_convert_example_to_features_init(self.slow_tokenizer)
        return path

    def process_inputs(self, inputs):
        for _ in range(10):
            start = time.perf_counter()
            super().process_inputs(inputs)
            new_dur = time.perf_counter() - start

            start = time.perf_counter()
            squad_example = SquadExample(
                inputs.id, inputs.question, inputs.context, "", None, None
            )
            features = squad_convert_example_to_features(
                example=squad_example,
                max_seq_length=self._sequence_length,
                doc_stride=self._doc_stride,
                max_query_length=self._max_question_length,
                is_training=False,
                padding_strategy="max_length",
            )
            old_dur = time.perf_counter() - start

            print("NEW:", new_dur * 1e3, "OLD:", old_dur * 1e3)
        return features

        # span_engine_inputs = []
        # span_extra_info = []
        # num_spans = len(tokenized_example["input_ids"])
        # for span in range(num_spans):
        #     span_input = [
        #         numpy.array(tokenized_example[key][span])
        #         for key in self.onnx_input_names
        #     ]
        #     span_engine_inputs.append(span_input)

        #     span_extra_info.append(
        #         {
        #             key: numpy.array(tokenized_example[key][span])
        #             for key in tokenized_example.keys()
        #             if key not in self.onnx_input_names
        #         }
        #     )

        # add batch dimension, assuming batch size 1
        # engine_inputs = list(map(numpy.stack, zip(*span_engine_inputs)))

        # return engine_inputs, dict(
        #     span_extra_info=span_extra_info, example=squad_example
        # )


def main():
    pipeline = QuestionAnsweringPipeline(
        model_path=PATH,
        version_2_with_negative=True,
        doc_stride=256,
        max_answer_length=512,
        max_question_length=512,
        sequence_length=512,
    )
    cuad = datasets.load_dataset("cuad")["test"]
    metric = CUAD()

    outputs = []

    for i in tqdm(range(1000)):
        example = cuad[i]

        output = pipeline(question=example["question"], context=example["context"])

        outputs.append((example, output))

        # print("=" * 50)
        # print(f"Answered: {output.answer}")
        # print()
        # print(f"Question: {example['question']}")
        # print()
        # print(f"Expected: {example['answers']}")
        # print()

    predictions = [
        {
            "id": example["id"],
            "prediction_text": [{"text": output.answer, "probability": 1.0}],
        }
        for example, output in outputs
    ]
    references = [
        {"id": example["id"], "answers": example["answers"]} for example, _ in outputs
    ]

    print(metric.compute(predictions=predictions, references=references))


if __name__ == "__main__":
    main()
