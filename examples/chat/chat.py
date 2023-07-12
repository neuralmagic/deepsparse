import argparse
from examples.codegen.text_generation import TextGenerationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a ChatGPT interactive interface."
    )
    parser.add_argument(
        "model_directory",
        type=str,
        help="Path to model directory containing model.onnx, config.json, and tokenizer.json",
    )
    args = parser.parse_args()

    print(f"Loading in model from {args.model_directory}")
    pipe = TextGenerationPipeline(
        model_path=args.model_directory,
        engine_type="onnxruntime",
        sequence_length=1024,
        num_tokens_to_generate=128,
        streamer=True,
        feature_size=64,
    )

    print("Welcome to the interactive ChatGPT interface!")
    print("Type 'exit' to end the session.")

    while True:
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Ending the session. Goodbye!")
            break

        result = pipe(sequences=user_input)
        print("Generated code:")
        print(result)


if __name__ == "__main__":
    main()
