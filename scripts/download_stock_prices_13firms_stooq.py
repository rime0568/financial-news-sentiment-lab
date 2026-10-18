import pandas as pd
from pathlib import Path
import time

TICKERS = [
    "TSLA", "JPM", "AAPL", "GOOG", "F", "NVDA", "UNH",
    "AMZN", "JNJ", "XOM", "BAC", "PFE", "CAT"
]

START_DATE = "20200101"
END_DATE = "20200715"

OUTPUT_FILE = Path("data/processed/stock_prices_13firms_2020H1.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker} from Stooq...")

    stooq_symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&d1={START_DATE}&d2={END_DATE}&i=d"

    try:
        df = pd.read_csv(url)

        if df.empty or "Date" not in df.columns:
            print(f"Warning: no data for {ticker}")
            continue

        df["ticker"] = ticker

        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        keep_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        df = df[keep_cols]

        all_data.append(df)

        print(f"Downloaded {ticker}: {len(df)} rows")

        time.sleep(1)

    except Exception as e:
        print(f"Failed {ticker}: {e}")

if len(all_data) == 0:
    print("No stock data downloaded. Please check network or data source.")
else:
    stock = pd.concat(all_data, ignore_index=True)
    stock = stock.sort_values(["ticker", "date"])
    stock.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nSaved to:", OUTPUT_FILE)
    print("Rows:", len(stock))
    print("Tickers:", stock["ticker"].nunique())
    print("Date range:", stock["date"].min(), "to", stock["date"].max())
    print("\nTicker counts:")
    print(stock["ticker"].value_counts())