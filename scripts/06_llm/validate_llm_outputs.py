"""Validate saved structured LLM outputs without treating confidence as truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VALID = {"positive", "neutral", "negative"}


def validate(input_path: Path, output_path: Path) -> None:
    rows = []
    for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
            label = item.get("sentiment")
            rows.append({"line_no": line_no, "valid_json": True, "valid_sentiment": label in VALID, "sentiment": label, "error": ""})
        except Exception as exc:
            rows.append({"line_no": line_no, "valid_json": False, "valid_sentiment": False, "sentiment": "", "error": repr(exc)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/model_evaluation/llm_output_validation.csv"))
    args = parser.parse_args()
    validate(args.input, args.output)
