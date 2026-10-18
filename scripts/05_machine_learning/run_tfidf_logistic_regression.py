"""Train a small chronological TF-IDF/logistic-regression sentiment baseline.

The script refuses to train when the human gold sample is incomplete. It uses
human labels only; dictionary or LLM labels are never silently substituted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = {"positive", "neutral", "negative"}


def run(input_path: Path, output_dir: Path, test_fraction: float) -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for this stage; install it in the V2 environment.") from exc

    df = pd.read_csv(input_path, low_memory=False)
    required = {"clean_text", "human_sentiment", "publication_datetime"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required gold-sample columns: {sorted(missing)}")
    df["human_sentiment"] = df["human_sentiment"].astype(str).str.lower().str.strip()
    if not df["human_sentiment"].isin(LABELS).all():
        counts = df["human_sentiment"].value_counts(dropna=False).to_dict()
        raise SystemExit(f"Gold sample is incomplete or has invalid labels: {counts}")
    if df["human_sentiment"].nunique() < 2:
        raise SystemExit("At least two human sentiment classes are required.")
    df["publication_datetime"] = pd.to_datetime(df["publication_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["publication_datetime", "clean_text"]).sort_values("publication_datetime")
    split = max(1, min(len(df) - 1, int(len(df) * (1 - test_fraction))))
    train, test = df.iloc[:split], df.iloc[split:]
    if train["human_sentiment"].nunique() < 2 or test["human_sentiment"].nunique() < 1:
        raise SystemExit("Chronological split does not contain usable classes; annotate more diverse dates.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train["clean_text"])
    x_test = vectorizer.transform(test["clean_text"])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, train["human_sentiment"])
    predicted = model.predict(x_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"metric": "train_rows", "value": len(train)},
            {"metric": "test_rows", "value": len(test)},
            {"metric": "accuracy", "value": accuracy_score(test["human_sentiment"], predicted)},
            {"metric": "macro_f1", "value": f1_score(test["human_sentiment"], predicted, average="macro", zero_division=0)},
        ]
    ).to_csv(output_dir / "tfidf_logistic_metrics.csv", index=False)
    report = classification_report(test["human_sentiment"], predicted, labels=sorted(LABELS), zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(output_dir / "tfidf_logistic_classification_report.csv")
    pd.DataFrame(
        confusion_matrix(test["human_sentiment"], predicted, labels=sorted(LABELS)),
        index=sorted(LABELS), columns=sorted(LABELS),
    ).to_csv(output_dir / "tfidf_logistic_confusion_matrix.csv")
    pd.DataFrame({"text": test["clean_text"], "human_sentiment": test["human_sentiment"], "predicted_sentiment": predicted}).to_csv(output_dir / "tfidf_logistic_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/interim/annotation/gold_sample_template_fnspid.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_evaluation/tfidf_logistic"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.test_fraction)
