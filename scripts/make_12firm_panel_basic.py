import pandas as pd
from pathlib import Path

NEWS_FILE = Path("data/processed/news_13firms_2020H1_pilot.csv")
STOCK_FILE = Path("data/processed/stock_prices_13firms_2020H1_from_fnspid.csv")

NEWS_OUTPUT = Path("data/processed/news_12firms_2020H1_pilot.csv")
PANEL_OUTPUT = Path("data/processed/firm_day_panel_12firms_2020H1_basic.csv")

# 12 matched firms: both news and stock price data are available
MATCHED_TICKERS = [
    "AAPL", "AMZN", "BAC", "CAT", "F", "GOOG",
    "JNJ", "JPM", "NVDA", "PFE", "TSLA", "XOM"
]

# =========================
# 1. Load data
# =========================

news = pd.read_csv(NEWS_FILE)
stock = pd.read_csv(STOCK_FILE)

news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
stock["ticker"] = stock["ticker"].astype(str).str.upper().str.strip()

news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.strftime("%Y-%m-%d")
stock["date"] = pd.to_datetime(stock["date"], errors="coerce").dt.strftime("%Y-%m-%d")

# =========================
# 2. Filter to 12 matched firms
# =========================

news_12 = news[news["ticker"].isin(MATCHED_TICKERS)].copy()
stock_12 = stock[stock["ticker"].isin(MATCHED_TICKERS)].copy()

news_12 = news_12.dropna(subset=["date", "ticker", "article_title"])
stock_12 = stock_12.dropna(subset=["date", "ticker"])

news_12 = news_12.sort_values(["ticker", "date"])
stock_12 = stock_12.sort_values(["ticker", "date"])

NEWS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
news_12.to_csv(NEWS_OUTPUT, index=False, encoding="utf-8-sig")

# =========================
# 3. Aggregate news to firm-day level
# =========================

news_daily = (
    news_12.groupby(["ticker", "date"])
    .agg(
        news_count=("article_title", "count"),
        headline_examples=("article_title", lambda x: " || ".join(x.dropna().astype(str).head(5)))
    )
    .reset_index()
)

news_daily["has_news"] = 1

# =========================
# 4. Merge stock data and news data
# =========================

panel = stock_12.merge(
    news_daily,
    on=["ticker", "date"],
    how="left"
)

panel["news_count"] = panel["news_count"].fillna(0).astype(int)
panel["has_news"] = panel["has_news"].fillna(0).astype(int)
panel["headline_examples"] = panel["headline_examples"].fillna("")

panel = panel.sort_values(["ticker", "date"])

panel.to_csv(PANEL_OUTPUT, index=False, encoding="utf-8-sig")

# =========================
# 5. Summary
# =========================

print("Saved 12-firm news sample to:", NEWS_OUTPUT)
print("Saved firm-day panel to:", PANEL_OUTPUT)

print("\nNews sample:")
print("Rows:", len(news_12))
print("Tickers:", news_12["ticker"].nunique())
print("Date range:", news_12["date"].min(), "to", news_12["date"].max())
print(news_12["ticker"].value_counts())

print("\nStock sample:")
print("Rows:", len(stock_12))
print("Tickers:", stock_12["ticker"].nunique())
print("Date range:", stock_12["date"].min(), "to", stock_12["date"].max())

print("\nFirm-day panel:")
print("Rows:", len(panel))
print("Tickers:", panel["ticker"].nunique())
print("Date range:", panel["date"].min(), "to", panel["date"].max())

print("\nNews-day coverage:")
print(panel["has_news"].value_counts())

print("\nTicker-level news days:")
print(panel.groupby("ticker")["has_news"].sum().sort_values(ascending=False))

print("\nFirst 10 rows:")
print(panel.head(10))