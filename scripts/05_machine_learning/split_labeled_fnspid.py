"""Filter usable FNSPID labels and create date-disjoint chronological splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID = {"positive", "neutral", "negative"}


def split(input_path: Path, output_path: Path, summary_path: Path, seed: int) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    df["human_sentiment"] = df["human_sentiment"].fillna("").astype(str).str.strip().str.lower()
    df["text_quality_flag"] = df["text_quality_flag"].fillna("").astype(str)
    usable = df.loc[
        df["source"].eq("fnspid")
        & ~df["text_quality_flag"].str.contains("possible_mojibake", regex=False)
        & df["human_sentiment"].isin(VALID)
        & df["clean_text"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    usable["publication_datetime"] = pd.to_datetime(usable["publication_datetime"], utc=True, errors="coerce")
    if usable["publication_datetime"].isna().any():
        raise SystemExit("Usable rows contain invalid publication dates; chronological splitting is unsafe.")
    if usable["clean_text"].duplicated().any():
        raise SystemExit("Exact duplicate texts must be resolved before chronological splitting.")

    usable["event_date"] = usable["publication_datetime"].dt.normalize()
    date_counts = usable.groupby("event_date", sort=True).size()
    cumulative = date_counts.cumsum()
    train_cutoff = (cumulative - len(usable) * 0.60).abs().idxmin()
    later_cumulative = cumulative.loc[cumulative.index > train_cutoff]
    if later_cumulative.empty:
        raise SystemExit("Not enough distinct publication dates for chronological validation and test splits.")
    validation_cutoff = (later_cumulative - len(usable) * 0.80).abs().idxmin()

    usable["split"] = "test"
    usable.loc[usable["event_date"] <= train_cutoff, "split"] = "train"
    usable.loc[(usable["event_date"] > train_cutoff) & (usable["event_date"] <= validation_cutoff), "split"] = "validation"
    for split_name, frame in usable.groupby("split"):
        if set(frame["human_sentiment"]) != VALID:
            raise SystemExit(f"Chronological {split_name} split does not contain all three labels.")

    split_rank = {"train": 0, "validation": 1, "test": 2}
    usable["split_rank"] = usable["split"].map(split_rank)
    usable = usable.sort_values(["split_rank", "publication_datetime", "annotation_id"]).reset_index(drop=True)
    usable.drop(columns=["event_date", "split_rank"], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(output_path, index=False, encoding="utf-8-sig")
    for split_name in ("train", "validation", "test"):
        usable.loc[usable["split"].eq(split_name)].to_csv(
            output_path.parent / f"fnspid_labeled_{split_name}.csv", index=False, encoding="utf-8-sig"
        )
    summary = usable.groupby(["split", "human_sentiment"], dropna=False).agg(
        rows=("annotation_id", "size"),
        start_date=("publication_datetime", "min"),
        end_date=("publication_datetime", "max"),
    ).reset_index()
    summary["split"] = pd.Categorical(summary["split"], categories=["train", "validation", "test"], ordered=True)
    summary = summary.sort_values(["split", "human_sentiment"])
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/interim/annotation/manual_annotations_fnspid.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/model_data/fnspid_labeled_split.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/model_evaluation/model_split_summary.csv"))
    parser.add_argument("--seed", type=int, default=20260816, help="Retained for CLI compatibility; chronological splitting does not randomise rows.")
    args = parser.parse_args()
    split(args.input, args.output, args.summary, args.seed)
