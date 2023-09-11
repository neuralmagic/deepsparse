from transformers import AutoTokenizer
from datasets import load_dataset


def process_concatenated_datasets(dataset_name, model_path, max_sequence_length, kwargs):
    if dataset_name == "wikitext2":
        eos = kwargs.get("eos", "\n\n")
        bos = kwargs.get("bos", "")

        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        raw_text = raw_dataset["text"]
    elif dataset_name == "c4":
        eos = kwargs.get("eos", "<|endoftext|>")
        bos = kwargs.get("bos", "")
        raw_samples = kwargs.get("raw_samples", None)
        data_file = kwargs.get("data_file", 0)
        if data_file is not None:
            raw_dataset = load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"validation": f"en/c4-validation.{data_file:05d}-of-00008.json.gz"},
                split="validation",
            )
        else:
            raw_dataset = load_dataset(
                "allenai/c4",
                "allenai--c4",
                split="validation",
            )
        if raw_samples is not None:
            raw_dataset = raw_dataset[:raw_samples]
        raw_text = raw_dataset["text"]

    # Dataset is split into sections that contain "max_sequence_length" tokens.
    # To split the dataset, first tokenize text
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return _split_text_by_tokens(
        raw_text, eos, bos, tokenizer, max_sequence_length,
    )


def _split_text_by_tokens(text, eos, bos, tokenizer, sequence_length):
    text = "".join([bos + sample + eos for sample in text])

    input_tokens = tokenizer(text, return_tensors="np",)[
        "input_ids"
    ][0]

    # Then split the tokenized text into sections of size "max_sequence_length" and
    # decode each section back into text format
    split_text = []
    for i in range(len(input_tokens) // sequence_length):
        start = i * sequence_length
        end = (i + 1) * sequence_length
        split_text.append(
            tokenizer.decode(
                input_tokens[start:end],
                clean_up_tokenization_spaces=False,
            )
        )

    # Handle any leftover tokens
    if (i + 1) * sequence_length < len(input_tokens):
        start = (i + 1) * sequence_length
        end = len(input_tokens)
        split_text.append(
            tokenizer.decode(
                input_tokens[start:end],
                clean_up_tokenization_spaces=False,
            )
        )

    return split_text