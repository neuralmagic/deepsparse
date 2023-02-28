from deepsparse.transformers import QuestionAnsweringPipeline
import datasets


pipeline = QuestionAnsweringPipeline(
    model_path="/nm/drive3/tuan/models/CUAD/sparse_transfer/potential/CUAD@oberta-pruned90@sparse_transfer_decay_cuad@EP10@BS32@H1.0@T5.0@WD0.0001@LR8e-5@ID20943/checkpoint-1536/deployment",
    version_2_with_negative=True,
    doc_stride=256,
    max_answer_length=512,
    max_question_length=512,
    sequence_length=512,
    num_cores=2,
)
cuad_test = datasets.load_dataset("cuad")["test"]

example = cuad_test[0]

output = pipeline(question=example["question"], context=example["context"])

print(example["question"])
print(output.answer)
print()
print("Correct answer:", example["answers"])
