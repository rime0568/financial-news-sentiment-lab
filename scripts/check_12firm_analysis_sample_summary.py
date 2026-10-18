from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/analysis_sample_12firms_2020H1.csv")
OUTPUT_FILE = Path("outputs/tables/analysis_sample_summary.csv")

REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "news_count",
    "log_news_count",
    "has_news",
    "has_sentiment",
    "headline_sentiment_mean",
    "headline_sentiment_mean_filled",
    "positive_headline_count",
    "negative_headline_count",
    "positive_headline_ratio",
    "negative_headline_ratio",
    "daily_return",
    "abnormal_volume",
    "next_day_return",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "downside_next_day",
    "large_downside_next_3day",
]

DESCRIBE_COLUMNS = [
    "news_count",
    "log_news_count",
    "headline_sentiment_mean",
    "headline_sentiment_mean_filled",
    "positive_headline_count",
    "negative_headline_count",
    "positive_headline_ratio",
    "negative_headline_ratio",
    "daily_return",
    "abnormal_volume",
    "next_day_return",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "downside_next_day",
    "large_downside_next_3day",
]

HAS_NEWS_MEAN_COLUMNS = [
    "next_day_return",
    "next_day_abs_return",
    "next_day_abnormal_volume",
    "downside_next_day",
    "large_downside_next_3day",
]


def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")


def check_required_columns(df):
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing_text}")


def add_summary_row(rows, section, item, metric, value):
    rows.append(
        {
            "section": section,
            "item": item,
            "metric": metric,
            "value": value,
        }
    )


check_input_file()

# =========================
# 1. Load data and check columns
# =========================

sample = pd.read_csv(INPUT_FILE)
check_required_columns(sample)

sample["ticker"] = sample["ticker"].astype(str).str.upper().str.strip()
sample["date"] = pd.to_datetime(sample["date"], errors="coerce")

for col in REQUIRED_COLUMNS:
    if col not in ["ticker", "date"]:
        sample[col] = pd.to_numeric(sample[col], errors="coerce")

summary_rows = []

# =========================
# 2. Basic sample information
# =========================

add_summary_row(summary_rows, "basic_info", "total_rows", "count", len(sample))
add_summary_row(summary_rows, "basic_info", "number_of_firms", "count", sample["ticker"].nunique())
add_summary_row(summary_rows, "basic_info", "date_range", "start", sample["date"].min().strftime("%Y-%m-%d"))
add_summary_row(summary_rows, "basic_info", "date_range", "end", sample["date"].max().strftime("%Y-%m-%d"))

for ticker, count in sample["ticker"].value_counts().sort_index().items():
    add_summary_row(summary_rows, "ticker_row_counts", ticker, "rows", int(count))

for value, count in sample["has_news"].value_counts(dropna=False).sort_index().items():
    add_summary_row(summary_rows, "has_news_distribution", f"has_news={int(value)}", "rows", int(count))

for value, count in sample["has_sentiment"].value_counts(dropna=False).sort_index().items():
    add_summary_row(summary_rows, "has_sentiment_distribution", f"has_sentiment={int(value)}", "rows", int(count))

# =========================
# 3. Descriptive statistics
# =========================

describe_stats = sample[DESCRIBE_COLUMNS].describe().T.reset_index()
describe_stats = describe_stats.rename(columns={"index": "item"})

for _, row in describe_stats.iterrows():
    item = row["item"]
    for metric in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        add_summary_row(summary_rows, "descriptive_statistics", item, metric, row[metric])

# =========================
# 4. Mean comparison by has_news
# =========================

has_news_means = (
    sample.groupby("has_news")[HAS_NEWS_MEAN_COLUMNS]
    .mean()
    .reset_index()
    .sort_values("has_news")
)

for _, row in has_news_means.iterrows():
    group_name = f"has_news={int(row['has_news'])}"
    for col in HAS_NEWS_MEAN_COLUMNS:
        add_summary_row(summary_rows, "mean_by_has_news", group_name, col, row[col])

# =========================
# 5. Ticker-level coverage
# =========================

ticker_summary = (
    sample.groupby("ticker")
    .agg(
        firm_days=("date", "size"),
        news_days=("has_news", "sum"),
        avg_news_count=("news_count", "mean"),
        avg_headline_sentiment=("headline_sentiment_mean", "mean"),
        avg_next_day_abs_return=("next_day_abs_return", "mean"),
        avg_next_day_abnormal_volume=("next_day_abnormal_volume", "mean"),
    )
    .reset_index()
    .sort_values("ticker")
)

for _, row in ticker_summary.iterrows():
    ticker = row["ticker"]
    for metric in [
        "firm_days",
        "news_days",
        "avg_news_count",
        "avg_headline_sentiment",
        "avg_next_day_abs_return",
        "avg_next_day_abnormal_volume",
    ]:
        add_summary_row(summary_rows, "ticker_coverage", ticker, metric, row[metric])

# =========================
# 6. Save output
# =========================

summary = pd.DataFrame(summary_rows)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# =========================
# 7. Print readable diagnostics
# =========================

print("Input file checked:", INPUT_FILE)
print("Output saved to:", OUTPUT_FILE)
print("Required columns present:", True)

print("\nBasic sample information:")
print("Total rows:", len(sample))
print("Number of firms:", sample["ticker"].nunique())
print("Date range:", sample["date"].min().strftime("%Y-%m-%d"), "to", sample["date"].max().strftime("%Y-%m-%d"))

print("\nRows by ticker:")
print(sample["ticker"].value_counts().sort_index())

print("\nhas_news distribution:")
print(sample["has_news"].value_counts(dropna=False).sort_index())

print("\nhas_sentiment distribution:")
print(sample["has_sentiment"].value_counts(dropna=False).sort_index())

print("\nCore variable descriptive statistics:")
print(sample[DESCRIBE_COLUMNS].describe().T)

print("\nMean outcomes by has_news:")
print(has_news_means.to_string(index=False))

print("\nTicker-level sample coverage:")
print(ticker_summary.to_string(index=False))
