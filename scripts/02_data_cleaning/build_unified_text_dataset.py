"""Build a provenance-preserving unified text table from FNSPID and web data."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


REQUIRED_WEB = {
    "source",
    "ticker",
    "company_name",
    "title",
    "publication_datetime",
    "url",
    "clean_text",
    "retrieved_at",
    "http_status",
    "content_hash",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_fnspid(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    title_col = "article_title" if "article_title" in raw.columns else "title"
    date_col = "date" if "date" in raw.columns else "Date"
    ticker_col = "ticker" if "ticker" in raw.columns else "Ticker"
    out = pd.DataFrame(
        {
            "source": "fnspid",
            "ticker": raw[ticker_col].astype(str).str.upper().str.strip(),
            "company_name": "",
            "title": raw[title_col].map(normalize_text),
            "publication_datetime": raw[date_col].astype(str),
            "url": raw["url"].astype(str) if "url" in raw.columns else "",
            "raw_text": raw[title_col].map(normalize_text),
            "clean_text": raw[title_col].map(normalize_text),
            "retrieved_at": "",
            "http_status": "",
            "content_hash": raw[title_col].map(normalize_text).map(
                lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
            ),
            "accession_number": "",
        }
    )
    return out


def load_web(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_WEB - set(raw.columns)
    if missing:
        raise ValueError(f"Web file is missing required columns: {sorted(missing)}")
    if "accession_number" not in raw.columns:
        raw["accession_number"] = ""
    return raw[
        [
            "source",
            "ticker",
            "company_name",
            "title",
            "publication_datetime",
            "url",
            "raw_text",
            "clean_text",
            "retrieved_at",
            "http_status",
            "content_hash",
            "accession_number",
        ]
    ].copy()


def build(fnspid_path: Path, web_path: Path, output_path: Path, quality_path: Path) -> None:
    fnspid = load_fnspid(fnspid_path)
    web = load_web(web_path)
    unified = pd.concat([fnspid, web], ignore_index=True)
    unified["ticker"] = unified["ticker"].astype(str).str.upper().str.strip()
    unified["clean_text"] = unified["clean_text"].map(normalize_text)
    unified["publication_datetime"] = pd.to_datetime(
        unified["publication_datetime"], errors="coerce", utc=True
    )
    unified["publication_date"] = unified["publication_datetime"].dt.date.astype("string")
    unified["text_length"] = unified["clean_text"].str.len()
    unified["is_valid_text"] = unified["text_length"].ge(10)
    unified["duplicate_key"] = unified["source"].astype(str) + "|" + unified["content_hash"].astype(str)
    unified["is_duplicate"] = unified.duplicated("duplicate_key", keep="first")
    duplicate_rows_removed = int(unified["is_duplicate"].sum())
    unified = unified.loc[~unified["is_duplicate"]].copy()
    unified["cross_source_duplicate"] = unified.duplicated("content_hash", keep=False)
    unified["analysis_eligible_default"] = ~unified["cross_source_duplicate"] & unified["is_valid_text"]
    unified.to_csv(output_path, index=False)

    quality = pd.DataFrame(
        [
            {"metric": "rows_before_dedup", "value": len(fnspid) + len(web)},
            {"metric": "rows_after_dedup", "value": len(unified)},
            {"metric": "fnspid_rows", "value": int((unified.source == "fnspid").sum())},
            {"metric": "web_rows", "value": int((unified.source != "fnspid").sum())},
            {"metric": "missing_publication_datetime", "value": int(unified.publication_datetime.isna().sum())},
            {"metric": "invalid_short_text", "value": int((~unified.is_valid_text).sum())},
            {"metric": "unique_tickers", "value": int(unified.ticker.nunique())},
            {"metric": "duplicate_rows_removed", "value": duplicate_rows_removed},
            {"metric": "cross_source_duplicate_rows_flagged", "value": int(unified["cross_source_duplicate"].sum())},
            {"metric": "default_analysis_eligible_rows", "value": int(unified["analysis_eligible_default"].sum())},
        ]
    )
    quality.to_csv(quality_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnspid", type=Path, default=Path("data/processed/news_12firms_2020H1_pilot.csv"))
    parser.add_argument("--web", type=Path, default=Path("data/interim/web_collected/sec_8k_small/sec_8k_supplement.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed_v2/unified_text_dataset.csv"))
    parser.add_argument("--quality-output", type=Path, default=Path("outputs/model_evaluation/unified_text_quality.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quality_output.parent.mkdir(parents=True, exist_ok=True)
    build(args.fnspid, args.web, args.output, args.quality_output)
