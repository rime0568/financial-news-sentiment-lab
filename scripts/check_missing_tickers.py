from pathlib import Path
import csv
import sys

INPUT_FILE = Path("data/raw/Stock_news/All_external.csv")

TARGET_TICKERS = {
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOG", "JPM", "BAC", "WMT",
    "XOM", "CVX", "JNJ", "PFE", "UNH",
    "BA", "CAT", "F", "GM", "NKE"
}

START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

# Allow large text fields
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

total_counts = {ticker: 0 for ticker in TARGET_TICKERS}
period_counts = {ticker: 0 for ticker in TARGET_TICKERS}
min_date = {ticker: None for ticker in TARGET_TICKERS}
max_date = {ticker: None for ticker in TARGET_TICKERS}

print("Input file exists:", INPUT_FILE.exists())
print("Input file:", INPUT_FILE)

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    date_idx = header.index("Date")
    ticker_idx = header.index("Stock_symbol")

    for i, row in enumerate(reader, start=1):
        if len(row) <= max(date_idx, ticker_idx):
            continue

        ticker = row[ticker_idx].strip().upper()
        if ticker not in TARGET_TICKERS:
            continue

        date = row[date_idx][:10]

        total_counts[ticker] += 1

        if min_date[ticker] is None or date < min_date[ticker]:
            min_date[ticker] = date

        if max_date[ticker] is None or date > max_date[ticker]:
            max_date[ticker] = date

        if START_DATE <= date <= END_DATE:
            period_counts[ticker] += 1

        if i % 1_000_000 == 0:
            print(f"Scanned rows: {i:,}", flush=True)

print("\nTicker coverage in raw file:")
print("Ticker | Total count | 2020-2023 count | Min date | Max date")
for ticker in sorted(TARGET_TICKERS):
    print(
        f"{ticker} | {total_counts[ticker]} | {period_counts[ticker]} | "
        f"{min_date[ticker]} | {max_date[ticker]}"
    )

missing_period = [t for t in sorted(TARGET_TICKERS) if period_counts[t] == 0]
print("\nMissing in 2020-2023:")
print(missing_period)
