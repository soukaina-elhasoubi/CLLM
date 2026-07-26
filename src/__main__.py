from llm_sdk.llm_sdk import Small_LLM_Model
from src.encoder import Encoder
from src.llm import LLM
from src.callmemaybe import CallMeMaybe
import json
import argparse

from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json'
    )
    parser.add_argument(
        '--input',
        default='data/input/function_calling_tests.json'
    )
    parser.add_argument(
        '--output',
        default='data/output/function_calling_results.json'
    )
    return parser.parse_args()


def create_encoder(vocab_path: str) -> Encoder:
    """Loads the tokenizer vocabulary and builds the encoder."""

    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        return Encoder(tokens)
    except Exception as e:
        print(f"Cannot load vocabulary: {e}")
        exit(1)


if __name__ == "__main__":
    try:
        args = parse_args()
        llm_model = Small_LLM_Model()
        encoder = create_encoder(llm_model.get_path_to_vocab_file())

        llm = LLM(llm_model, encoder)
        cmm = CallMeMaybe(llm, args.functions_definition)

        with open(args.input) as requests:
            data = json.load(requests)

        if not isinstance(data, list):
            raise ValueError(
                "function_calling_tests must contain a list"
            )

        if not data:
            raise ValueError(
                "function_calling_tests cannot be empty"
            )

        for item in data:
            if (
                not isinstance(item, dict)
                or "prompt" not in item
                or not isinstance(item["prompt"], str)
            ):
                raise ValueError(
                    "Each test must contain a string 'prompt'"
                )

        prompts = [item["prompt"] for item in data]

        Path(args.output).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        results = []

        for p in prompts:
            results.append(cmm.process_func(p))

        with open(args.output, "w") as output:
            output.write("[\n")
            output.write(",\n".join(results))
            output.write("\n]")

        print('Finished.')

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        exit(1)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e.msg} " +
              f"at line {e.lineno} column {e.colno}")
        exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        exit(1)
