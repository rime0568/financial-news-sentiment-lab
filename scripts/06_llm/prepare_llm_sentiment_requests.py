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


def make_requests(input_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    required = {"clean_text", "human_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if not df["human_sentiment"].astype(str).str.strip().isin({"positive", "neutral", "negative"}).all():
        raise SystemExit("Human labels are incomplete; do not prepare evaluated LLM requests yet.")
    output_dir.mkdir(parents=True, exist_ok=True)
    few_shot = df.head(6)[["clean_text", "human_sentiment"]].to_dict("records")
    with (output_dir / "llm_zero_shot_requests.jsonl").open("w", encoding="utf-8") as zero, (output_dir / "llm_few_shot_requests.jsonl").open("w", encoding="utf-8") as few:
        for idx, row in df.iterrows():
            base = {
                "request_id": int(idx),
                "model": "CONFIGURE_BEFORE_RUN",
                "prompt_version": "sentiment_v1",
                "input": row["clean_text"],
                "system": SYSTEM,
            }
            zero.write(json.dumps({**base, "mode": "zero_shot"}, ensure_ascii=False) + "\n")
            examples = "\n".join(f"Text: {x['clean_text']}\nLabel: {x['human_sentiment']}" for x in few_shot)
            few.write(json.dumps({**base, "mode": "few_shot", "examples": examples}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/interim/annotation/gold_sample_template_fnspid.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/llm_requests"))
    args = parser.parse_args()
    make_requests(args.input, args.output_dir)
