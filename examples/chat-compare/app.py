import argparse
from transformers import TextStreamer
from deepsparse import Pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Simulate an interactive text-generation interface to evaluate a model."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to SparseZoo stub or model directory containing model.onnx, config.json, and tokenizer.json",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()

    # Construct pipelines
    ds_pipe = Pipeline.create(
        task="text-generation",
        model_path=args.model_path,
        max_generated_tokens=args.max_new_tokens,
        prompt_processing_sequence_length=1,
        use_deepsparse_cache=False,
        engine_type="deepsparse"
    )
    ort_pipe = Pipeline.create(
        task="text-generation",
        model_path=args.model_path,
        max_generated_tokens=args.max_new_tokens,
        prompt_processing_sequence_length=1,
        use_deepsparse_cache=False,
        engine_type="onnxruntime"
    )

    print("Welcome to the interactive text generation interface!")
    print("Type 'exit' to end the session.")

    while True:
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Ending the session. Goodbye!")
            break

        streamer = TextStreamer(ds_pipe.tokenizer)

        print("DeepSparse output:")
        _ = ds_pipe(sequences=user_input, streamer=streamer)

        print("ORT output:")
        _ = ort_pipe(sequences=user_input, streamer=streamer)
        


if __name__ == "__main__":
    main()