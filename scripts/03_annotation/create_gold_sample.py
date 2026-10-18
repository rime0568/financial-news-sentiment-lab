"""Create a reproducible, stratified human-annotation template."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def create(input_path: Path, output_path: Path, n: int, seed: int, source: str | None) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    df = df.loc[df["is_valid_text"].astype(bool)].copy()
    if source:
        df = df.loc[df["source"].eq(source)].copy()
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["source_ticker_group"] = df["source"].astype(str) + "|" + df["ticker"].astype(str)
    groups = df["source_ticker_group"].dropna().unique().tolist()
    per_group = max(1, n // max(1, len(groups)))
    parts = []
    for group in groups:
        part = df.loc[df["source_ticker_group"] == group].sample(
            n=min(per_group, int((df["source_ticker_group"] == group).sum())),
            random_state=seed,
        )
        parts.append(part)
    sample = pd.concat(parts, ignore_index=True).drop_duplicates("duplicate_key")
    if len(sample) < n:
        remaining = df.loc[~df["duplicate_key"].isin(sample["duplicate_key"])].sample(
            n=min(n - len(sample), len(df) - len(sample)), random_state=seed
        )
        sample = pd.concat([sample, remaining], ignore_index=True)
    sample = sample.head(n).copy()
    sample.insert(0, "annotation_id", range(1, len(sample) + 1))
    sample["human_sentiment"] = ""
    sample["annotation_confidence"] = ""
    sample["annotation_note"] = ""
    columns = [
        "annotation_id",
        "source",
        "ticker",
        "company_name",
        "title",
        "clean_text",
        "publication_datetime",
        "url",
        "accession_number",
        "human_sentiment",
        "annotation_confidence",
        "annotation_note",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(output_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed_v2/unified_text_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/annotation/gold_sample_template_fnspid.csv"))
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--source", default="fnspid", help="Source to annotate; use empty string for all sources.")
    args = parser.parse_args()
    create(args.input, args.output, args.n, args.seed, args.source or None)
