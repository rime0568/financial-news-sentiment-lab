import re
from pathlib import Path

import pandas as pd


NEWS_FILE = Path("data/processed/news_12firms_2020H1_pilot.csv")
PANEL_FILE = Path("data/processed/firm_day_panel_12firms_2020H1_basic.csv")
OUTPUT_FILE = Path("data/processed/firm_day_panel_12firms_2020H1_with_headline_sentiment.csv")


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
    "plunged", "pressure", "pressured", "risk", "risks", "sell", "sells",
    "selling", "slump", "slumps", "slumped", "spill", "spills", "tumble",
    "tumbles", "tumbled", "weak", "weaker", "weakness", "worry", "worries",
    "worried", "worse", "worst",
}


def tokenize_headline(text):
    if pd.isna(text):
        return []
    return re.findall(r"[a-z]+", str(text).lower())


def count_words(tokens, word_set):
    return sum(1 for token in tokens if token in word_set)


def check_input_files():
    missing_files = [path for path in [NEWS_FILE, PANEL_FILE] if not path.exists()]
    if missing_files:
        missing_text = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing input file(s): {missing_text}")


check_input_files()

# =========================
# 1. Load data
# =========================

news = pd.read_csv(NEWS_FILE)
panel = pd.read_csv(PANEL_FILE)

news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()

news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.strftime("%Y-%m-%d")
panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.strftime("%Y-%m-%d")

news = news.dropna(subset=["date", "ticker", "article_title"]).copy()
panel = panel.dropna(subset=["date", "ticker"]).copy()

# =========================
# 2. Article-level headline sentiment
# =========================

news["headline_tokens"] = news["article_title"].apply(tokenize_headline)
news["headline_word_count"] = news["headline_tokens"].apply(len)
news["positive_word_count"] = news["headline_tokens"].apply(lambda x: count_words(x, POSITIVE_WORDS))
news["negative_word_count"] = news["headline_tokens"].apply(lambda x: count_words(x, NEGATIVE_WORDS))
news["headline_sentiment"] = news["positive_word_count"] - news["negative_word_count"]

news["positive_headline"] = (news["headline_sentiment"] > 0).astype(int)
news["negative_headline"] = (news["headline_sentiment"] < 0).astype(int)
news["neutral_headline"] = (news["headline_sentiment"] == 0).astype(int)

# =========================
# 3. Aggregate to ticker-date
# =========================

sentiment_daily = (
    news.groupby(["ticker", "date"])
    .agg(
        headline_sentiment_mean=("headline_sentiment", "mean"),
        headline_sentiment_sum=("headline_sentiment", "sum"),
        headline_sentiment_max=("headline_sentiment", "max"),
        headline_sentiment_min=("headline_sentiment", "min"),
        positive_headline_count=("positive_headline", "sum"),
        negative_headline_count=("negative_headline", "sum"),
        neutral_headline_count=("neutral_headline", "sum"),
        headline_word_count_mean=("headline_word_count", "mean"),
    )
    .reset_index()
)

# =========================
# 4. Same-date merge into firm-day panel
# =========================

panel_with_sentiment = panel.merge(
    sentiment_daily,
    on=["ticker", "date"],
    how="left",
)

panel_with_sentiment = panel_with_sentiment.sort_values(["ticker", "date"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
panel_with_sentiment.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# =========================
# 5. Summary
# =========================

sentiment_cols = [
    "headline_sentiment_mean",
    "headline_sentiment_sum",
    "headline_sentiment_max",
    "headline_sentiment_min",
    "positive_headline_count",
    "negative_headline_count",
    "neutral_headline_count",
    "headline_word_count_mean",
]

firm_days_with_sentiment = panel_with_sentiment["headline_sentiment_mean"].notna().sum()

print("Saved panel with headline sentiment to:", OUTPUT_FILE)

print("\nInput files checked:")
print("News file exists:", NEWS_FILE.exists(), "-", NEWS_FILE)
print("Panel file exists:", PANEL_FILE.exists(), "-", PANEL_FILE)

print("\nArticle-level headline sentiment:")
print("News rows used:", len(news))
print("Positive headlines:", int(news["positive_headline"].sum()))
print("Negative headlines:", int(news["negative_headline"].sum()))
print("Neutral headlines:", int(news["neutral_headline"].sum()))

print("\nFirm-day panel with sentiment:")
print("Rows:", len(panel_with_sentiment))
print("Tickers:", panel_with_sentiment["ticker"].nunique())
print("Date range:", panel_with_sentiment["date"].min(), "to", panel_with_sentiment["date"].max())
print("Firm-days with sentiment data:", int(firm_days_with_sentiment))
print("Firm-days without sentiment data:", int(len(panel_with_sentiment) - firm_days_with_sentiment))

print("\nSentiment columns added:")
print(sentiment_cols)

print("\nFirst 10 rows with sentiment columns:")
print(panel_with_sentiment[["date", "ticker", "news_count"] + sentiment_cols].head(10))
