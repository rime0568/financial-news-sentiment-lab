import pandas as pd
from pathlib import Path

# =========================
# 1. Basic settings
# =========================

INPUT_FILE = Path("data/raw/Stock_news/All_external.csv")
OUTPUT_FILE = Path("data/processed/news_20_companies_2020_2023.csv")

TARGET_TICKERS = {
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOG", "JPM", "BAC", "WMT",
    "XOM", "CVX", "JNJ", "PFE", "UNH",
    "BA", "CAT", "F", "GM", "NKE"
}

START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

CHUNKSIZE = 100_000

USECOLS = [
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Publisher",
    "Author",
    "Article",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
]

# =========================
# 2. Prepare output folder
# =========================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()
    print(f"Removed old output file: {OUTPUT_FILE}")

print("Input file exists:", INPUT_FILE.exists())
print("Input file:", INPUT_FILE)
print("Output file:", OUTPUT_FILE)

# =========================
# 3. Read large CSV by chunks
# =========================

total_rows = 0
kept_rows = 0
first_write = True

for chunk_id, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=USECOLS,
        chunksize=CHUNKSIZE,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
    ),
    start=1,
):
    total_rows += len(chunk)

    # Standardize ticker
    chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper().str.strip()

    # Parse date
    chunk["datetime"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
    chunk = chunk.dropna(subset=["datetime"])

    # Create date and month
    chunk["date"] = chunk["datetime"].dt.date.astype(str)
    chunk["month"] = chunk["datetime"].dt.to_period("M").astype(str)

    # Filter ticker and date range
    filtered = chunk[
        (chunk["Stock_symbol"].isin(TARGET_TICKERS))
        & (chunk["date"] >= START_DATE)
        & (chunk["date"] <= END_DATE)
    ].copy()

    if len(filtered) > 0:
        filtered = filtered.rename(
            columns={
                "Stock_symbol": "ticker",
                "Article_title": "article_title",
                "Url": "url",
                "Publisher": "publisher",
                "Author": "author",
                "Article": "article",
                "Lsa_summary": "lsa_summary",
                "Luhn_summary": "luhn_summary",
                "Textrank_summary": "textrank_summary",
                "Lexrank_summary": "lexrank_summary",
            }
        )

        output_cols = [
            "date",
            "month",
            "ticker",
            "article_title",
            "url",
            "publisher",
            "author",
            "article",
            "lsa_summary",
            "luhn_summary",
            "textrank_summary",
            "lexrank_summary",
        ]

        filtered[output_cols].to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig",
        )

        first_write = False
        kept_rows += len(filtered)

    if chunk_id % 20 == 0:
        print(
            f"Processed chunks: {chunk_id}, "
            f"total rows scanned: {total_rows:,}, "
            f"rows kept: {kept_rows:,}"
        )

print("\nFinished filtering.")
print(f"Total rows scanned: {total_rows:,}")
print(f"Rows kept: {kept_rows:,}")
print(f"Saved to: {OUTPUT_FILE}")