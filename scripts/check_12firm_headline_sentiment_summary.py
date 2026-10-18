from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/firm_day_panel_12firms_2020H1_with_headline_sentiment.csv")
OUTPUT_DIR = Path("outputs/tables")
SUMMARY_BY_TICKER_FILE = OUTPUT_DIR / "sentiment_summary_by_ticker.csv"
EXTREME_FIRM_DAYS_FILE = OUTPUT_DIR / "extreme_sentiment_firm_days.csv"

REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "news_count",
    "headline_examples",
    "has_news",
    "headline_sentiment_mean",
    "headline_sentiment_sum",
    "positive_headline_count",
    "negative_headline_count",
    "neutral_headline_count",
    "headline_word_count_mean",
]

DESCRIBE_COLUMNS = [
    "headline_sentiment_mean",
    "headline_sentiment_sum",
    "positive_headline_count",
    "negative_headline_count",
    "neutral_headline_count",
    "headline_word_count_mean",
]

EXTREME_COLUMNS = [
    "ticker",
    "date",
    "headline_examples",
    "headline_sentiment_mean",
    "news_count",
]


def check_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")


def check_required_columns(df):
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing_text}")


check_input_file()

# =========================
# 1. Load data and check columns
# =========================

panel = pd.read_csv(INPUT_FILE)
check_required_columns(panel)

panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.strftime("%Y-%m-%d")

for col in DESCRIBE_COLUMNS + ["news_count", "has_news"]:
    panel[col] = pd.to_numeric(panel[col], errors="coerce")

# =========================
# 2. Overall descriptive statistics
# =========================

overall_stats = panel[DESCRIBE_COLUMNS].describe()

# =========================
# 3. Ticker-level summary
# =========================

summary_by_ticker = (
    panel.groupby("ticker")
    .agg(
        firm_days=("date", "size"),
        news_days=("headline_sentiment_mean", lambda x: x.notna().sum()),
        avg_headline_sentiment=("headline_sentiment_mean", "mean"),
        total_positive_headlines=("positive_headline_count", "sum"),
        total_negative_headlines=("negative_headline_count", "sum"),
        total_neutral_headlines=("neutral_headline_count", "sum"),
        total_news_count=("news_count", "sum"),
    )
    .reset_index()
    .sort_values(["news_days", "ticker"], ascending=[False, True])
)

count_columns = [
    "firm_days",
    "news_days",
    "total_positive_headlines",
    "total_negative_headlines",
    "total_neutral_headlines",
    "total_news_count",
]
summary_by_ticker[count_columns] = summary_by_ticker[count_columns].astype(int)

# =========================
# 4. Extreme sentiment firm-days
# =========================

sentiment_days = panel[panel["headline_sentiment_mean"].notna()].copy()

top_positive = (
    sentiment_days.sort_values(
        ["headline_sentiment_mean", "news_count", "ticker", "date"],
        ascending=[False, False, True, True],
    )
    .head(10)
    .copy()
)
top_positive["extreme_type"] = "most_positive"

top_negative = (
    sentiment_days.sort_values(
        ["headline_sentiment_mean", "news_count", "ticker", "date"],
        ascending=[True, False, True, True],
    )
    .head(10)
    .copy()
)
top_negative["extreme_type"] = "most_negative"

extreme_firm_days = pd.concat([top_positive, top_negative], ignore_index=True)
extreme_firm_days = extreme_firm_days[["extreme_type"] + EXTREME_COLUMNS]

# =========================
# 5. Save summary tables
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_by_ticker.to_csv(SUMMARY_BY_TICKER_FILE, index=False, encoding="utf-8-sig")
extreme_firm_days.to_csv(EXTREME_FIRM_DAYS_FILE, index=False, encoding="utf-8-sig")

# =========================
# 6. Print summary
# =========================

print("Input file checked:", INPUT_FILE)
print("Rows:", len(panel))
print("Columns:", len(panel.columns))
print("Required columns present:", True)

print("\nOverall descriptive statistics:")
print(overall_stats)

print("\nTicker-level sentiment summary:")
print(summary_by_ticker.to_string(index=False))

print("\nTop 10 most positive firm-days:")
print(top_positive[EXTREME_COLUMNS].to_string(index=False))

print("\nTop 10 most negative firm-days:")
print(top_negative[EXTREME_COLUMNS].to_string(index=False))

print("\nSaved ticker summary to:", SUMMARY_BY_TICKER_FILE)
print("Saved extreme sentiment firm-days to:", EXTREME_FIRM_DAYS_FILE)
