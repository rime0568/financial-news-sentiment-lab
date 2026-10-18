"""Smoke-test chronological splitting and leakage-safe LLM request preparation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def run(*parts: str) -> None:
    subprocess.run([sys.executable, *parts], cwd=ROOT, check=True)


with tempfile.TemporaryDirectory() as folder:
    temp = Path(folder)
    rows = []
    labels = ("negative", "neutral", "positive")
    for day in range(1, 11):
        for label in labels:
            rows.append({
                "annotation_id": len(rows) + 1,
                "source": "fnspid",
                "ticker": "DEMO",
                "human_sentiment": label,
                "clean_text": f"Synthetic {label} financial headline number {day}",
                "publication_datetime": f"2020-01-{day:02d} 00:00:00+00:00",
                "text_quality_flag": "ok",
            })
    source = temp / "synthetic_labels.csv"
    split_file = temp / "split.csv"
    summary = temp / "summary.csv"
    llm_dir = temp / "llm"
    pd.DataFrame(rows).to_csv(source, index=False)

    run("scripts/05_machine_learning/split_labeled_fnspid.py", "--input", str(source), "--output", str(split_file), "--summary", str(summary))
    split = pd.read_csv(split_file)
    split["publication_datetime"] = pd.to_datetime(split["publication_datetime"], utc=True)
    assert split.loc[split["split"].eq("train"), "publication_datetime"].max() < split.loc[split["split"].eq("validation"), "publication_datetime"].min()
    assert split.loc[split["split"].eq("validation"), "publication_datetime"].max() < split.loc[split["split"].eq("test"), "publication_datetime"].min()

    run("scripts/06_llm/prepare_llm_sentiment_requests.py", "--input", str(split_file), "--output-dir", str(llm_dir))
    requests = [json.loads(line) for line in (llm_dir / "llm_few_shot_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    test_ids = set(split.loc[split["split"].eq("test"), "annotation_id"].astype(int))
    assert {item["request_id"] for item in requests} == test_ids
    assert all(item["evaluation_split"] == "test" for item in requests)

print("Financial NLP smoke tests passed.")
