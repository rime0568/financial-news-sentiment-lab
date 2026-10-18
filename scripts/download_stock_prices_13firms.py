import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = [
    "TSLA", "JPM", "AAPL", "GOOG", "F", "NVDA", "UNH",
    "AMZN", "JNJ", "XOM", "BAC", "PFE", "CAT"
]

START_DATE = "2020-01-01"
END_DATE = "2020-07-15"

OUTPUT_FILE = Path("data/processed/stock_prices_13firms_2020H1.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        print(f"Warning: no data for {ticker}")
        continue

    df = df.reset_index()
    df["ticker"] = ticker

    # Flatten columns if yfinance returns multi-index columns
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    keep_cols = ["Date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[keep_cols]

    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    all_data.append(df)

stock = pd.concat(all_data, ignore_index=True)

stock = stock.sort_values(["ticker", "date"])
stock.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nSaved to:", OUTPUT_FILE)
print("Rows:", len(stock))
print("Tickers:", stock["ticker"].nunique())
print("Date range:", stock["date"].min(), "to", stock["date"].max())
print(stock.head())