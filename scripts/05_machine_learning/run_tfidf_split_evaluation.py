"""Evaluate TF-IDF logistic regression using the frozen V2 data split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = ["negative", "neutral", "positive"]


def metrics(y_true, y_pred, split_name: str, accuracy_score, f1_score):
    return {
        "split": split_name,
        "rows": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }


def run(input_path: Path, output_dir: Path) -> None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise SystemExit("scikit-learn is required; run this script in an environment with scikit-learn installed.") from exc

    df = pd.read_csv(input_path, low_memory=False)
    df["human_sentiment"] = df["human_sentiment"].fillna("").astype(str).str.strip().str.lower()
    if not set(df["split"]).issubset({"train", "validation", "test"}):
        raise SystemExit("Input split column contains unexpected values.")
    train = df[df["split"].eq("train")]
    validation = df[df["split"].eq("validation")]
    test = df[df["split"].eq("test")]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_train = vectorizer.fit_transform(train["clean_text"])
    x_validation = vectorizer.transform(validation["clean_text"])
    scores = []
    best_c, best_score = None, -1.0
    for c_value in (0.1, 1.0, 10.0):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=c_value)
        model.fit(x_train, train["human_sentiment"])
        pred = model.predict(x_validation)
        score = f1_score(validation["human_sentiment"], pred, labels=LABELS, average="macro", zero_division=0)
        scores.append({"C": c_value, "validation_macro_f1": score})
        if score > best_score:
            best_c, best_score = c_value, score

    train_validation = pd.concat([train, validation], ignore_index=True)
    final_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_train_validation = final_vectorizer.fit_transform(train_validation["clean_text"])
    x_test = final_vectorizer.transform(test["clean_text"])
    final_model = LogisticRegression(max_iter=1000, class_weight="balanced", C=best_c)
    final_model.fit(x_train_validation, train_validation["human_sentiment"])
    test_pred = final_model.predict(x_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(output_dir / "tfidf_c_selection.csv", index=False)
    pd.DataFrame([metrics(test["human_sentiment"], test_pred, "test_final", accuracy_score, f1_score)]).to_csv(output_dir / "tfidf_metrics.csv", index=False)
    pd.DataFrame(
        classification_report(test["human_sentiment"], test_pred, labels=LABELS, zero_division=0, output_dict=True)
    ).T.to_csv(output_dir / "tfidf_test_classification_report.csv")
    pd.DataFrame(confusion_matrix(test["human_sentiment"], test_pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(output_dir / "tfidf_test_confusion_matrix.csv")
    pd.DataFrame(
        {
            "annotation_id": test["annotation_id"],
            "text": test["clean_text"],
            "human_sentiment": test["human_sentiment"],
            "predicted_sentiment": test_pred,
        }
    ).to_csv(output_dir / "tfidf_test_predictions.csv", index=False)
    pd.DataFrame(
        [{"metric": "selected_C", "value": best_c}, {"metric": "validation_macro_f1_at_selected_C", "value": best_score}]
    ).to_csv(output_dir / "tfidf_run_metadata.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/interim/model_data/fnspid_labeled_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_evaluation/tfidf_logistic_split"))
    args = parser.parse_args()
    run(args.input, args.output_dir)
