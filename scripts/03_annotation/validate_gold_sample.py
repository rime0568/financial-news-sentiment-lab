"""Validate completion and class balance of the human gold sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID = {"positive", "neutral", "negative"}


def validate(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    labels = df["human_sentiment"].fillna("").astype(str).str.lower().str.strip()
    valid = labels.isin(VALID)
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "labeled_rows", "value": int(valid.sum())},
            {"metric": "unlabeled_rows", "value": int((~valid).sum())},
            {"metric": "invalid_label_rows", "value": int((labels.ne("") & ~valid).sum())},
            {"metric": "positive_rows", "value": int((labels == "positive").sum())},
            {"metric": "neutral_rows", "value": int((labels == "neutral").sum())},
            {"metric": "negative_rows", "value": int((labels == "negative").sum())},
            {"metric": "annotation_status", "value": "complete" if valid.all() and labels.nunique() == 3 else "incomplete_or_invalid"},
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    if not (valid.all() and labels.nunique() == 3):
        raise SystemExit("Gold sample is not complete with all three valid classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/interim/annotation/gold_sample_template_fnspid.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model_evaluation/gold_sample_validation.csv"))
    args = parser.parse_args()
    validate(args.input, args.output)
