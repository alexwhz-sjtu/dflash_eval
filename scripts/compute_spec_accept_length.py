from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("response_longbench_v2_dflash_full.json")


def to_float(value, field_name: str, index: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} at record {index}: {value!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute weighted average spec_accept_length by model invocation count: "
            "sum(spec_verify_ct * spec_accept_length) / sum(spec_verify_ct)."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include records marked with warmup=true. By default they are skipped.",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{args.input} must contain a JSON list")

    weighted_sum = 0.0
    total_verify_ct = 0.0
    used_records = 0
    skipped_records = 0

    for index, record in enumerate(records):
        if record.get("warmup") and not args.include_warmup:
            skipped_records += 1
            continue

        meta = record.get("meta_info") or {}
        if "spec_verify_ct" not in meta or "spec_accept_length" not in meta:
            skipped_records += 1
            continue

        spec_verify_ct = to_float(meta.get("spec_verify_ct"), "spec_verify_ct", index)
        spec_accept_length = to_float(
            meta.get("spec_accept_length"), "spec_accept_length", index
        )
        if spec_verify_ct <= 0:
            skipped_records += 1
            continue

        weighted_sum += spec_verify_ct * spec_accept_length
        total_verify_ct += spec_verify_ct
        used_records += 1

    if total_verify_ct <= 0:
        raise RuntimeError("No records with positive spec_verify_ct were found.")

    average = weighted_sum / total_verify_ct
    print(f"input: {args.input}")
    print(f"used_records: {used_records}")
    print(f"skipped_records: {skipped_records}")
    print(f"total_spec_verify_ct: {total_verify_ct:.0f}")
    print(f"weighted_spec_accept_length: {average:.6f}")


if __name__ == "__main__":
    main()
