"""Filter usable FNSPID labels and create deterministic train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
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
    usable["text_group"] = usable["clean_text"].map(lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest())
    # Keep exact duplicate texts together. Groups are shuffled deterministically
    # within each class, then allocated approximately 60/20/20.
    usable["split"] = ""
    for label, idx in usable.groupby("human_sentiment", sort=True).groups.items():
        groups = usable.loc[idx].groupby("text_group").size().sample(frac=1, random_state=seed).index.tolist()
        target_test = round(len(idx) * 0.20)
        target_validation = round(len(idx) * 0.20)
        test_rows = validation_rows = 0
        test_groups, validation_groups = set(), set()
        for group in groups:
            size = int((usable.loc[idx, "text_group"] == group).sum())
            if test_rows < target_test:
                test_groups.add(group)
                test_rows += size
            elif validation_rows < target_validation:
                validation_groups.add(group)
                validation_rows += size
        usable.loc[usable["text_group"].isin(test_groups), "split"] = "test"
        usable.loc[usable["text_group"].isin(validation_groups), "split"] = "validation"
        usable.loc[(usable["human_sentiment"] == label) & usable["split"].eq(""), "split"] = "train"
    usable.drop(columns=["text_group"], inplace=True)
    usable = usable.sort_values(["split", "annotation_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    usable.to_csv(output_path, index=False, encoding="utf-8-sig")
    for split_name in ("train", "validation", "test"):
        usable.loc[usable["split"].eq(split_name)].to_csv(
            output_path.parent / f"fnspid_labeled_{split_name}.csv", index=False, encoding="utf-8-sig"
        )
    summary = (
        usable.groupby(["split", "human_sentiment"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["split", "human_sentiment"])
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/interim/annotation/manual_annotations_fnspid.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/model_data/fnspid_labeled_split.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/model_evaluation/model_split_summary.csv"))
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    split(args.input, args.output, args.summary, args.seed)
