import argparse
import json
import random
import string

from deepsparse.pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSparse Pipelines"
    )
    parser.add_argument(
        "task_name",
        type=str
    )
    parser.add_argument(
        "model_path",
        type=str
    )
    parser.add_argument(
        "-i",
        "--input_type",
        type=str,
        default="dummy",
        choices=["dummy", "real"],
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.json",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    config_file = open(args.config)
    config = json.load(config_file)
    config_file.close()

    task_name = args.task_name
    model_path = args.model_path

    data_length = config['length']
    num_examples = config['num_examples']
    examples = []
    if config['input_data_type'] == "string":
        for _ in range(num_examples):
            rand_string = ''.join(random.choices(string.printable, k=data_length))
            examples.append(rand_string)
    print(examples)

    pipeline = Pipeline.create(task=task_name, model_path=model_path)
    output = pipeline(examples)
    print(output)
    print(pipeline.timer_manger)
    print(pipeline.timer_manager.stages)

if __name__ == "__main__":
    main()