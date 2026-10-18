"""Collect a small, compliant SEC EDGAR 8-K text supplement.

This script is intentionally conservative: it uses the SEC submissions API,
sets an identifying User-Agent, throttles requests, retries transient failures,
and writes provenance-rich records. It does not modify the FNSPID dataset or
feed records into the original market-reaction panel automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


COMPANIES = {
    "AAPL": ("Apple Inc.", "0000320193"),
    "AMZN": ("Amazon.com, Inc.", "0001018724"),
    "BAC": ("Bank of America Corporation", "0000070858"),
    "CAT": ("Caterpillar Inc.", "0000018230"),
    "F": ("Ford Motor Company", "0000037996"),
    "GOOG": ("Alphabet Inc.", "0001652044"),
    "JNJ": ("Johnson & Johnson", "0000200406"),
    "JPM": ("JPMorgan Chase & Co.", "0000019617"),
    "NVDA": ("NVIDIA Corporation", "0001045810"),
    "PFE": ("Pfizer Inc.", "0000078003"),
    "TSLA": ("Tesla, Inc.", "0001318605"),
    "XOM": ("Exxon Mobil Corporation", "0000034088"),
}


@dataclass
class FetchResult:
    status: int | None
    text: str
    error: str | None = None


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def request_with_retry(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> FetchResult:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                return FetchResult(response.status_code, response.text)
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = repr(exc)
        if attempt < retries:
            time.sleep(sleep_seconds * (2**attempt))
    return FetchResult(None, "", last_error or "unknown error")


def submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


def filing_url(cik: str, accession: str, primary_document: str) -> str:
    accession_no_dash = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_no_dash}/{primary_document}"
    )


def iter_recent_8k_filings(payload: dict[str, Any], limit: int) -> Iterable[dict[str, Any]]:
    recent = payload.get("filings", {}).get("recent", {})
    rows = zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("primaryDocument", []),
    )
    count = 0
    for form, accession, filing_date, report_date, primary_document in rows:
        if form != "8-K":
            continue
        yield {
            "accession_number": accession,
            "filing_date": filing_date,
            "report_date": report_date,
            "primary_document": primary_document,
        }
        count += 1
        if count >= limit:
            break


def collect(output_dir: Path, tickers: list[str], per_ticker: int, sleep_seconds: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "FinancialNewsResearch/1.0 research-contact@example.invalid",
            "Accept-Encoding": "gzip, deflate",
        }
    )

    for ticker in tickers:
        company, cik = COMPANIES[ticker]
        url = submissions_url(cik)
        result = request_with_retry(session, url, timeout=30, retries=2, sleep_seconds=sleep_seconds)
        if result.error:
            errors.append({"ticker": ticker, "stage": "submissions", "url": url, "error": result.error})
            continue
        payload = json.loads(result.text)
        for filing in iter_recent_8k_filings(payload, per_ticker):
            document_url = filing_url(cik, filing["accession_number"], filing["primary_document"])
            time.sleep(sleep_seconds)
            document = request_with_retry(
                session,
                document_url,
                timeout=30,
                retries=2,
                sleep_seconds=sleep_seconds,
            )
            retrieved_at = datetime.now(timezone.utc).isoformat()
            if document.error:
                errors.append({"ticker": ticker, "stage": "document", "url": document_url, "error": document.error})
                continue
            raw_name = f"{ticker}_{filing['accession_number'].replace('-', '')}.html"
            (raw_dir / raw_name).write_text(document.text, encoding="utf-8")
            text = clean_text(document.text)
            records.append(
                {
                    "source": "sec_edgar_8k",
                    "ticker": ticker,
                    "company_name": company,
                    "title": filing["primary_document"],
                    "publication_datetime": filing["filing_date"],
                    "report_date": filing["report_date"],
                    "url": document_url,
                    "accession_number": filing["accession_number"],
                    "raw_text": document.text,
                    "clean_text": text,
                    "retrieved_at": retrieved_at,
                    "http_status": document.status,
                    "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )

    pd.DataFrame(records).to_csv(output_dir / "sec_8k_supplement.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "collection_errors.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "tickers_requested", "value": len(tickers)},
            {"metric": "documents_requested_per_ticker", "value": per_ticker},
            {"metric": "documents_downloaded", "value": len(records)},
            {"metric": "errors", "value": len(errors)},
            {"metric": "duplicate_content_hashes", "value": len(records) - len({r["content_hash"] for r in records})},
        ]
    ).to_csv(output_dir / "collection_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/web_collected/sec_8k_small"))
    parser.add_argument("--per-ticker", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--tickers", nargs="+", default=list(COMPANIES))
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    unknown = sorted(set(args.tickers) - set(COMPANIES))
    if unknown:
        raise SystemExit(f"Unknown ticker(s): {', '.join(unknown)}")
    collect(args.output_dir, args.tickers, args.per_ticker, args.sleep_seconds)
