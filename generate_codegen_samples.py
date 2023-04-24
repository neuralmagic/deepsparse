import argparse
import os
import numpy as np
import onnx
import onnxruntime as rt
from transformers import AutoTokenizer
from collections import defaultdict
from functools import partial


def main(args: argparse.Namespace):
    if not os.path.exists(args.onnx_export_dir):
        raise ValueError("The directory {} does not exist".format(args.onnx_export_dir))

    # Load the model, session and set up the tokenizer
    multitoken_onnx_path = os.path.join(args.onnx_export_dir, "decoder_model.onnx")
    single_token_onnx_path = os.path.join(args.onnx_export_dir, "decoder_with_past_model.onnx")

    sess_multitoken = rt.InferenceSession(multitoken_onnx_path)
    sess_single_token = rt.InferenceSession(single_token_onnx_path)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")

    output_names = [output.name for output in onnx.load(single_token_onnx_path).graph.output]

    # Prepare the input data
    model_input = tokenizer(args.input_sequence, return_tensors='np').data
    attention_mask = model_input["attention_mask"]

    # Prepare placeholders to collect the ground truth
    generated_sequence = []
    numpy_data = defaultdict(partial(defaultdict))

    # Run the model
    # Assuming we are in the 0th iteration
    out = sess_multitoken.run(output_names, model_input)
    logits, *kv_cache = out

    # Assuming that the multitoken run is a 0th iteration
    numpy_data[0]['inputs'] = model_input
    numpy_data[0]['outputs'] = out

    for i in range(1, args.num_iterations + 1):
        # transform the "out" kv cache dictionary to "input" kv cache dictionary
        kv_cache = {k.replace("present", "past_key_values"): v for k, v in zip(output_names[1:], kv_cache)}
        # get the predicted token
        tok = np.argmax(logits[0, -1])
        # add it to the generated sequence
        generated_sequence.append(tok)
        # prepare the input for the next iteration
        input_ids = np.array([[tok]])
        attention_mask = np.hstack((attention_mask, np.ones((1, 1)))).astype(np.int64)

        model_input = {"input_ids": input_ids, "attention_mask": attention_mask, **kv_cache}
        out = sess_single_token.run(output_names, model_input)
        logits, *kv_cache = out

        numpy_data[i]['inputs'] = model_input
        numpy_data[i]['outputs'] = out

    # save numpy_data to a file
    np.savez(args.output_file, numpy_data)
    print("Saved the generated sequence to {}".format(args.output_file))
    print(f"Final result:\n{args.input_sequence + tokenizer.decode(generated_sequence, skip_special_tokens=True)}")


if "__main__" == __name__:
    """
    Example usage:
    1) Export the onnx models from optimum: 
        optimum-cli export onnx --model Salesforce/codegen-350M-multi codegen-350M-multi
    2) Run the script:
        python run_onnx.py -o codegen-350M-multi -i "def hello_world():" -n 100 -f numpy_data.npz
    3) 
       Inspect the data saved with the following structure:
       epoch 0:
           inputs: {input_ids: ..., attention_mask: ..., present_key_values: ...}
           outputs: [logits, past_key_values]
       epoch 1:
           inputs: {input_ids: ..., attention_mask: ..., present_key_values: ...}
           outputs: [logits, past_key_values]
        ...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_export_dir", "-o", required=True, type=str,
                        help="The directory that contains the exported ONNX models from optimum")
    parser.add_argument("--input_sequence", "-i", default="def hello_world():", type=str,
                        help="Input sequence to be used to generate the output")
    parser.add_argument("--num_iterations", "-n", default=100, type=int,
                        help="Number of autoregressive iterations to be used to generate the output")
    parser.add_argument("--output_file", "-f", default="numpy_data.npz", type=str,
                        help="Output file to save the generated sequence")
    args = parser.parse_args()
    main(args)
