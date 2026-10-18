from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/firm_day_panel_12firms_2020H1_with_headline_sentiment.csv")
OUTPUT_FILE = Path("data/processed/analysis_sample_12firms_2020H1.csv")

CORE_COLUMNS = [
    "ticker",
    "date",
    "close",
    "volume",
    "daily_return",
    "abnormal_volume",
    "next_day_return",
    "next_3day_return",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "downside_next_day",
    "large_downside_next_3day",
    "news_count",
    "has_news",
    "headline_sentiment_mean",
    "headline_sentiment_sum",
    "positive_headline_count",
    "negative_headline_count",
    "neutral_headline_count",
    "headline_word_count_mean",
]

KEY_MARKET_REACTION_COLUMNS = [
    "next_day_return",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "daily_return",
]


def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")


def check_required_columns(df):
    missing_columns = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing_text}")


check_input_file()

# =========================
# 1. Load data and check columns
# =========================

panel = pd.read_csv(INPUT_FILE)
check_required_columns(panel)

original_rows = len(panel)

analysis_sample = panel[CORE_COLUMNS].copy()
analysis_sample["ticker"] = analysis_sample["ticker"].astype(str).str.upper().str.strip()
analysis_sample["date"] = pd.to_datetime(analysis_sample["date"], errors="coerce").dt.strftime("%Y-%m-%d")

numeric_columns = [col for col in CORE_COLUMNS if col not in ["ticker", "date"]]
for col in numeric_columns:
    analysis_sample[col] = pd.to_numeric(analysis_sample[col], errors="coerce")

# =========================
# 2. Create analysis variables
# =========================

analysis_sample["news_count"] = analysis_sample["news_count"].fillna(0).astype(int)
analysis_sample["has_news"] = analysis_sample["has_news"].fillna(0).astype(int)

analysis_sample["has_sentiment"] = analysis_sample["headline_sentiment_mean"].notna().astype(int)
analysis_sample["log_news_count"] = np.log1p(analysis_sample["news_count"])

count_fill_zero_columns = [
    "positive_headline_count",
    "negative_headline_count",
    "neutral_headline_count",
    "headline_sentiment_sum",
]

for col in count_fill_zero_columns:
    analysis_sample[col] = analysis_sample[col].fillna(0)

analysis_sample["headline_sentiment_mean_filled"] = analysis_sample["headline_sentiment_mean"].fillna(0)

analysis_sample["negative_headline_ratio"] = np.where(
    analysis_sample["news_count"] > 0,
    analysis_sample["negative_headline_count"] / analysis_sample["news_count"],
    0,
)
analysis_sample["positive_headline_ratio"] = np.where(
    analysis_sample["news_count"] > 0,
    analysis_sample["positive_headline_count"] / analysis_sample["news_count"],
    0,
)

# =========================
# 3. Drop rows with missing key market reaction variables
# =========================

missing_before_drop = analysis_sample.isna().sum()

analysis_sample_clean = analysis_sample.dropna(subset=KEY_MARKET_REACTION_COLUMNS).copy()
analysis_sample_clean = analysis_sample_clean.sort_values(["ticker", "date"])

clean_rows = len(analysis_sample_clean)
deleted_rows = original_rows - clean_rows

# =========================
# 4. Save output
# =========================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
analysis_sample_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# =========================
# 5. Diagnostics
# =========================

key_missing_before_drop = missing_before_drop[
    KEY_MARKET_REACTION_COLUMNS
    + [
        "news_count",
        "headline_sentiment_mean",
        "headline_sentiment_mean_filled",
        "headline_sentiment_sum",
        "positive_headline_count",
        "negative_headline_count",
        "positive_headline_ratio",
        "negative_headline_ratio",
    ]
]

key_missing_after_drop = analysis_sample_clean[
    KEY_MARKET_REACTION_COLUMNS
    + [
        "news_count",
        "headline_sentiment_mean",
        "headline_sentiment_mean_filled",
        "headline_sentiment_sum",
        "positive_headline_count",
        "negative_headline_count",
        "positive_headline_ratio",
        "negative_headline_ratio",
    ]
].isna().sum()

print("Input file checked:", INPUT_FILE)
print("Output saved to:", OUTPUT_FILE)

print("\nRows:")
print("Original rows:", original_rows)
print("Clean rows:", clean_rows)
print("Deleted rows:", deleted_rows)

print("\nRows by ticker:")
print(analysis_sample_clean["ticker"].value_counts().sort_index())

print("\nhas_news distribution:")
print(analysis_sample_clean["has_news"].value_counts(dropna=False).sort_index())

print("\nhas_sentiment distribution:")
print(analysis_sample_clean["has_sentiment"].value_counts(dropna=False).sort_index())

print("\nMissing values before dropping key market reaction rows:")
print(key_missing_before_drop)

print("\nMissing values after dropping key market reaction rows:")
print(key_missing_after_drop)

print("\nFirst 10 rows:")
print(analysis_sample_clean.head(10))
