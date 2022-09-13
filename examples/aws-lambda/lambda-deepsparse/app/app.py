import json
from deepsparse import Pipeline

qa_pipeline = Pipeline.create(task="question-answering", model_path="./model/deployment")

def lambda_handler(event, context):

    body = json.loads(event["body"])
    question = body["question"]
    context = body["context"]

    inference = qa_pipeline(question=question, context=context)
    print(f"Question: {question}, Answer: {inference.answer}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "Question": question,
            "Answer": inference.answer
        })
    }
