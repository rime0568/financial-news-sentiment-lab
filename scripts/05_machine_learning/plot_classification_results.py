"""Create a compact confusion-matrix figure from saved aggregate results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(confusion_path: Path, output_path: Path) -> None:
    matrix = pd.read_csv(confusion_path, index_col=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, linewidths=1, linecolor="white", ax=ax)
    ax.set_title("TF-IDF Logistic Regression: Chronological Holdout", pad=16, weight="bold")
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Human-reviewed sentiment")
    fig.text(0.5, 0.01, "53 articles · 2020-05-14 to 2020-06-10 · Macro-F1 0.511", ha="center", color="#52656a")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confusion", type=Path, default=Path("results/metrics/tfidf_test_confusion_matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/tfidf_chronological_confusion_matrix.png"))
    args = parser.parse_args()
    plot(args.confusion, args.output)
