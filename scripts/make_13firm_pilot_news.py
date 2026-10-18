import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/news_20_companies_2020_2023.csv")
OUTPUT_FILE = Path("data/processed/news_13firms_2020H1_pilot.csv")

PILOT_TICKERS = [
    "TSLA", "JPM", "AAPL", "GOOG", "F", "NVDA", "UNH",
    "AMZN", "JNJ", "XOM", "BAC", "PFE", "CAT"
]

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

pilot = df[df["ticker"].isin(PILOT_TICKERS)].copy()
pilot = pilot.dropna(subset=["date", "ticker", "article_title"])

pilot["date"] = pilot["date"].dt.strftime("%Y-%m-%d")
pilot["month"] = pd.to_datetime(pilot["date"]).dt.to_period("M").astype(str)

pilot = pilot.sort_values(["ticker", "date"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
pilot.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Saved to:", OUTPUT_FILE)
print("Rows:", len(pilot))
print("Tickers:", pilot["ticker"].nunique())
print("Date range:", pilot["date"].min(), "to", pilot["date"].max())
print("\nTicker counts:")
print(pilot["ticker"].value_counts())