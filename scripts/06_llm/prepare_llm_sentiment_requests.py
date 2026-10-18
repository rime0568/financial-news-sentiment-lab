"""Prepare auditable zero-shot and few-shot LLM request records.

This script does not call an external provider and does not read API keys. It
creates JSONL requests that can later be executed through an approved client.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM = (
    "Classify the financial text using only the text provided. Return JSON only "
    "with one field: sentiment, whose value is positive, neutral, or negative."
)


def make_requests(input_path: Path, output_dir: Path, evaluation_split: str, examples_per_class: int) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    required = {"annotation_id", "clean_text", "human_sentiment", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if not df["human_sentiment"].astype(str).str.strip().isin({"positive", "neutral", "negative"}).all():
        raise SystemExit("Human labels are incomplete; do not prepare evaluated LLM requests yet.")
    if evaluation_split not in {"validation", "test"}:
        raise SystemExit("Evaluation split must be validation or test.")
    training = df.loc[df["split"].eq("train")].copy()
    evaluation = df.loc[df["split"].eq(evaluation_split)].copy()
    if evaluation.empty:
        raise SystemExit(f"No rows found for evaluation split: {evaluation_split}")
    few_shot = (
        training.sort_values("annotation_id")
        .groupby("human_sentiment", group_keys=False)
        .head(examples_per_class)[["annotation_id", "clean_text", "human_sentiment"]]
    )
    if few_shot.groupby("human_sentiment").size().min() < examples_per_class:
        raise SystemExit("Training split does not contain enough examples for every class.")
    if set(few_shot["annotation_id"]) & set(evaluation["annotation_id"]):
        raise SystemExit("Few-shot examples overlap the evaluation split.")

    output_dir.mkdir(parents=True, exist_ok=True)
    examples = few_shot.to_dict("records")
    with (output_dir / "llm_zero_shot_requests.jsonl").open("w", encoding="utf-8") as zero, (output_dir / "llm_few_shot_requests.jsonl").open("w", encoding="utf-8") as few:
        for _, row in evaluation.iterrows():
            base = {
                "request_id": int(row["annotation_id"]),
                "model": "CONFIGURE_BEFORE_RUN",
                "prompt_version": "sentiment_v1",
                "input": row["clean_text"],
                "system": SYSTEM,
                "evaluation_split": evaluation_split,
            }
            zero.write(json.dumps({**base, "mode": "zero_shot"}, ensure_ascii=False) + "\n")
            formatted_examples = "\n".join(f"Text: {x['clean_text']}\nLabel: {x['human_sentiment']}" for x in examples)
            few.write(json.dumps({**base, "mode": "few_shot", "examples": formatted_examples}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/interim/model_data/fnspid_labeled_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/llm_requests"))
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="test")
    parser.add_argument("--examples-per-class", type=int, default=2)
    args = parser.parse_args()
    make_requests(args.input, args.output_dir, args.evaluation_split, args.examples_per_class)
