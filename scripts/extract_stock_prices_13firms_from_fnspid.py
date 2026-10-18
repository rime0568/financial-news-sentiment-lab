import pandas as pd
import numpy as np
from pathlib import Path

STOCK_DIR = Path("data/raw/Stock_price/full_history/full_history")
OUTPUT_FILE = Path("data/processed/stock_prices_13firms_2020H1_from_fnspid.csv")

TICKERS = [
    "TSLA", "JPM", "AAPL", "GOOG", "F", "NVDA", "UNH",
    "AMZN", "JNJ", "XOM", "BAC", "PFE", "CAT"
]

START_DATE = "2020-01-01"
END_DATE = "2020-07-15"

all_data = []

for ticker in TICKERS:
    file_path = STOCK_DIR / f"{ticker}.csv"

    if not file_path.exists():
        print(f"Missing file for {ticker}: {file_path}")
        continue

    print(f"Reading {ticker}...")

    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "date" not in df.columns:
        print(f"Warning: no date column for {ticker}. Columns: {df.columns.tolist()}")
        continue

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[
        (df["date"] >= START_DATE)
        & (df["date"] <= END_DATE)
    ].copy()

    if df.empty:
        print(f"Warning: no 2020H1 data for {ticker}")
        continue

    df["ticker"] = ticker

    # Make numeric columns safe
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols].copy()

    all_data.append(df)

if len(all_data) == 0:
    raise ValueError("No stock data extracted.")

stock = pd.concat(all_data, ignore_index=True)
stock = stock.sort_values(["ticker", "date"])

# Construct market variables
stock["daily_return"] = stock.groupby("ticker")["close"].pct_change()
stock["log_return"] = stock.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
stock["abs_return"] = stock["daily_return"].abs()

# Intraday range proxy for volatility
stock["range_volatility"] = (stock["high"] - stock["low"]) / stock["close"]

# Abnormal volume: current volume relative to previous 20 trading days
stock["volume_ma20"] = (
    stock.groupby("ticker")["volume"]
    .transform(lambda x: x.shift(1).rolling(window=20, min_periods=5).mean())
)

stock["abnormal_volume"] = stock["volume"] / stock["volume_ma20"] - 1

# Future market reaction variables
stock["next_day_return"] = stock.groupby("ticker")["close"].shift(-1) / stock["close"] - 1
stock["next_3day_return"] = stock.groupby("ticker")["close"].shift(-3) / stock["close"] - 1
stock["next_day_abs_return"] = stock["next_day_return"].abs()
stock["next_day_volume"] = stock.groupby("ticker")["volume"].shift(-1)
stock["next_day_abnormal_volume"] = stock.groupby("ticker")["abnormal_volume"].shift(-1)

# Downside risk dummy
stock["downside_next_day"] = (stock["next_day_return"] < 0).astype(int)
stock["large_downside_next_3day"] = (stock["next_3day_return"] <= -0.03).astype(int)

stock["date"] = stock["date"].dt.strftime("%Y-%m-%d")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
stock.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nSaved to:", OUTPUT_FILE)
print("Rows:", len(stock))
print("Tickers:", stock["ticker"].nunique())
print("Date range:", stock["date"].min(), "to", stock["date"].max())

print("\nTicker counts:")
print(stock["ticker"].value_counts())

print("\nFirst 5 rows:")
print(stock.head())