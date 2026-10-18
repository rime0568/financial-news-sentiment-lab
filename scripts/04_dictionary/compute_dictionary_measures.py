"""Compute transparent dictionary measures on the unified V2 text table.

The original project dictionary is included as a baseline. An optional
Loughran–McDonald CSV can be supplied explicitly; the script never invents
financial-dictionary labels when that file is absent.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


POSITIVE_WORDS = {
    "beat", "beats", "beating", "bull", "bullish", "buy", "buys", "gain", "gains",
    "gained", "gaining", "grow", "grows", "growth", "grew", "higher", "high",
    "jump", "jumps", "jumped", "jumping", "outperform", "outperforms", "positive",
    "profit", "profits", "profitable", "raise", "raises", "raised", "raising",
    "rally", "rallies", "rallied", "rebound", "rebounds", "record", "recover",
    "recovers", "recovered", "strong", "stronger", "strength", "surge", "surges",
    "surged", "up", "upgrade", "upgrades", "upgraded", "win", "wins", "winner",
}
NEGATIVE_WORDS = {
    "bear", "bearish", "cut", "cuts", "cutting", "decline", "declines", "declined",
    "downgrade", "downgrades", "downgraded", "drop", "drops", "dropped", "dropping",
    "fall", "falls", "fell", "falling", "fear", "fears", "loss", "losses", "lost",
    "low", "lower", "miss", "misses", "missed", "negative", "plunge", "plunges",
    "plunged", "pressure", "pressured", "risk", "risks", "sell", "sells", "selling",
    "slump", "slumps", "slumped", "spill", "spills", "tumble", "tumbles", "tumbled",
    "weak", "weaker", "weakness", "worry", "worries", "worried", "worse", "worst",
}


def tokens(text: object) -> list[str]:
    return re.findall(r"[a-z]+", "" if pd.isna(text) else str(text).lower())


def load_lm(path: Path | None) -> dict[str, set[str]] | None:
    if path is None or not path.exists():
        return None
    table = pd.read_csv(path)
    word_col = next((c for c in table.columns if c.lower() in {"word", "words"}), None)
    if word_col is None:
        raise ValueError("LM file must contain a Word or Words column")
    normalized = {c.lower(): c for c in table.columns}
    output: dict[str, set[str]] = {}
    for label in ("positive", "negative", "uncertainty", "litigious", "constraining"):
        column = normalized.get(label)
        if column is None:
            output[label] = set()
            continue
        values = table.loc[table[column].fillna(0).astype(bool), word_col]
        output[label] = {str(value).lower().strip() for value in values}
    return output


def compute(input_path: Path, output_path: Path, status_path: Path, lm_path: Path | None) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    df["tokens"] = df["clean_text"].map(tokens)
    df["word_count"] = df["tokens"].str.len()
    df["baseline_positive_count"] = df["tokens"].map(lambda x: sum(t in POSITIVE_WORDS for t in x))
    df["baseline_negative_count"] = df["tokens"].map(lambda x: sum(t in NEGATIVE_WORDS for t in x))
    denominator = df["word_count"].replace(0, pd.NA)
    df["baseline_positive_ratio"] = df["baseline_positive_count"] / denominator
    df["baseline_negative_ratio"] = df["baseline_negative_count"] / denominator
    df["baseline_sentiment"] = df["baseline_positive_count"] - df["baseline_negative_count"]

    lm = load_lm(lm_path)
    if lm is None:
        for label in ("positive", "negative", "uncertainty", "litigious", "constraining"):
            df[f"lm_{label}_count"] = pd.NA
            df[f"lm_{label}_ratio"] = pd.NA
        lm_status = "not_run_missing_explicit_dictionary_file"
    else:
        for label, words in lm.items():
            df[f"lm_{label}_count"] = df["tokens"].map(lambda x, words=words: sum(t in words for t in x))
            df[f"lm_{label}_ratio"] = df[f"lm_{label}_count"] / denominator
        lm_status = "computed"

    df.drop(columns=["tokens"], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    pd.DataFrame(
        [
            {"method": "original_baseline_dictionary", "status": "computed", "source": "original project word sets"},
            {"method": "loughran_mcdonald", "status": lm_status, "source": str(lm_path) if lm_path else "no explicit file supplied"},
        ]
    ).to_csv(status_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed_v2/unified_text_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed_v2/unified_text_with_dictionary_measures.csv"))
    parser.add_argument("--status-output", type=Path, default=Path("outputs/model_evaluation/dictionary_status.csv"))
    parser.add_argument("--lm-file", type=Path, default=None)
    args = parser.parse_args()
    compute(args.input, args.output, args.status_output, args.lm_file)
