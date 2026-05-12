from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("/data/wanghanzhen/datasets/LongBench_v2/data.json")
DEFAULT_TOKENIZER = "/data/wanghanzhen/models/Qwen/Qwen3-8B"


def build_prompt(item: dict) -> str:
    if "context" not in item:
        raise ValueError("Missing 'context' field")
    if "question" not in item:
        raise ValueError("Missing 'question' field")
    return f"{item['context']}\n\nQuestion: {item['question']}"


def drop_choice_fields(item: dict) -> dict:
    return {
        key: value
        for key, value in item.items()
        if key != "choice" and not key.startswith("choice_")
    }


def write_json_array_item(f, item: dict, *, first_written: bool) -> None:
    if not first_written:
        f.write(",\n")
    item_json = json.dumps(item, ensure_ascii=False, indent=2)
    f.write("  " + item_json.replace("\n", "\n  "))
    f.flush()


def iter_with_progress(data: list, total: int):
    try:
        from tqdm import tqdm

        yield from tqdm(data, desc="Filtering LongBench_v2", unit="item")
        return
    except ImportError:
        pass

    for index, item in enumerate(data, start=1):
        print(f"Filtering LongBench_v2: {index}/{total}", end="\r", flush=True)
        yield item
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter LongBench_v2 examples by Qwen3-tokenized prompt length. "
            "The prompt is context + question, and choice fields are removed."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minlength", type=int, required=True)
    parser.add_argument("--maxlength", type=int, required=True)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.from_pretrained.",
    )
    args = parser.parse_args()

    if args.minlength < 0:
        raise ValueError("--minlength must be non-negative")
    if args.maxlength < args.minlength:
        raise ValueError("--maxlength must be greater than or equal to --minlength")

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{args.input} must contain a JSON list")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    too_short = 0
    too_long = 0
    kept = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("[\n")
        f.flush()
        first_written = True
        for item in iter_with_progress(data, len(data)):
            prompt = build_prompt(item)
            token_length = len(tokenizer.encode(prompt, add_special_tokens=False))
            if token_length < args.minlength:
                too_short += 1
                continue
            if token_length > args.maxlength:
                too_long += 1
                continue

            output_item = drop_choice_fields(item)
            output_item["length"] = token_length
            write_json_array_item(f, output_item, first_written=first_written)
            first_written = False
            kept += 1
        f.write("\n]\n")

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"minlength: {args.minlength}")
    print(f"maxlength: {args.maxlength}")
    print(f"total: {len(data)}")
    print(f"kept: {kept}")
    print(f"too_short: {too_short}")
    print(f"too_long: {too_long}")


if __name__ == "__main__":
    main()
