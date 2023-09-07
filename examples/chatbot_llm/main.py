import argparse
from deepsparse import Pipeline

def run(args):

    pipeline = Pipeline.create(
        task="text-generation",
        model_path=args.model_path,
        max_generated_tokens = args.max_generated_tokens,
    )

    while True:
        # get input from user
        input_text = input("User: ")
        response = pipeline(sequences=[input_text], session_ids=args.session_id)
        print("Bot: ", response.sequences[0])


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model directory or sparsezoo stub")
    parser.add_argument("-s", "--session_id", help="Name of the session that will be used in the example", default="session_1")
    parser.add_argument("-m", "--max_generated_tokens", help="Max number of tokens to generate", default=32, type=int)
    args = parser.parse_args()
    run(args)