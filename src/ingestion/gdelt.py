"""Fetch recent global news from the GDELT DOC 2.0 API into the common schema.

GDELT monitors world news and updates every 15 minutes. The DOC API is free and
needs no API key. It covers roughly the last 3 months, returns at most 250
articles per request, and rate-limits heavy use (officially one request every
5 seconds).

GDELT matches words anywhere on a web page, including menus and share buttons,
so many results are not really about the company. We therefore keep only
articles whose HEADLINE mentions the company (or, for market-wide queries, a
macro / geopolitical keyword). This is our entity-linking step for news.

Each run merges new articles into data/raw/gdelt.jsonl. That file is committed
to the repo so demo mode works offline.

Run from the repo root:  python -m src.ingestion.gdelt
"""
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.ingestion.tweets import COLUMNS, clean_text

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_CSV = ROOT / "data" / "universe.csv"
CACHE_FILE = ROOT / "data" / "raw" / "gdelt.jsonl"
API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

REQUEST_GAP_SECONDS = 10         # pause between queries
BACKOFF_SECONDS = [20, 45]       # waits before retrying a rate-limited query
RETRY_PASS_PAUSE_SECONDS = 90    # pause before a final retry of failed queries

# Company queries must also contain a finance word.
FINANCE_TERMS = "(stock OR shares OR earnings OR investors OR analysts)"

# Market-wide queries -> ticker "MARKET". They feed the Macroeconomic and
# Geopolitical event classes and the Module B stress test.
# Each entry: (GDELT query, words at least one of which must be in the headline)
MARKET_QUERIES = [
    (
        '("Federal Reserve" OR "interest rates" OR inflation OR recession OR "central bank")',
        ["Federal Reserve", "Fed", "interest rate", "rate hike", "rate cut", "inflation",
         "recession", "central bank", "CPI", "jobs report", "GDP"],
    ),
    (
        '(sanctions OR tariffs OR "trade war" OR invasion OR "military conflict") (markets OR stocks OR investors)',
        ["sanction", "tariff", "trade war", "invasion", "military", "conflict", "war",
         "missile", "ceasefire", "geopolitical"],
    ),
]

# GDELT rejects search words shorter than 5 characters ("Meta", "AMD", "AWS")
# and some symbols ("P&G"). Those are still used for the headline filter.
GDELT_SAFE = re.compile(r"^[A-Za-z0-9 .\-]{5,}$")


class GdeltError(Exception):
    pass


class RateLimited(GdeltError):
    pass


def build_company_query(aliases, company_name):
    """'Apple|iPhone' -> ("Apple" OR "iPhone") (stock OR shares OR ...)"""
    terms = [a.strip() for a in aliases.split("|") if GDELT_SAFE.match(a.strip())]
    if not terms:
        terms = [company_name.split()[0]]  # e.g. Procter & Gamble -> "Procter"
    names = " OR ".join(f'"{t}"' for t in terms)
    if len(terms) > 1:
        names = f"({names})"  # GDELT only allows brackets around OR'd terms
    return f"{names} {FINANCE_TERMS}"


def build_headline_filter(terms):
    """Regex that matches any of the terms as a whole word in a headline.

    Short terms with capitals (AMD, AWS, P&G, Meta, Fed, CPI) are case-sensitive,
    so "meta" or "fed up" don't count. Longer terms ignore case and allow a
    plural ending ("tariff" also matches "tariffs").
    """
    parts = []
    for term in terms:
        term = term.strip()
        escaped = re.escape(term)
        if len(term) <= 4 and any(c.isupper() for c in term):
            parts.append(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
        else:
            parts.append(rf"(?i:(?<![A-Za-z0-9]){escaped}(?:s|es)?(?![A-Za-z0-9]))")
    return re.compile("|".join(parts))


def fetch_query(query, timespan):
    """Run one GDELT ArtList search and return the raw list of articles."""
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 250,
        "sort": "DateDesc",
        "timespan": timespan,
    }
    headers = {"User-Agent": "risk-engine-hackathon/1.0"}

    for wait in BACKOFF_SECONDS + [None]:
        resp = requests.get(API_URL, params=params, headers=headers, timeout=60)
        text = resp.text.strip()

        if "limit requests" in text.lower() or resp.status_code == 429:
            if wait is None:
                raise RateLimited("rate limited")
            time.sleep(wait)
            continue
        if not text:
            return []
        try:
            data = json.loads(text, strict=False)  # tolerates odd characters in titles
        except json.JSONDecodeError:
            raise GdeltError(text[:200])  # GDELT replies in plain text when it rejects a query
        return data.get("articles", [])


def articles_to_rows(articles, ticker, headline_filter):
    """Keep articles whose headline passes the filter; convert to the common schema."""
    rows = []
    for a in articles:
        title = clean_text(a.get("title") or "")
        url = a.get("url")
        if not title or not url or not headline_filter.search(title):
            continue
        rows.append({
            "id": "gd_" + hashlib.sha1(f"{ticker}|{url}".encode("utf-8")).hexdigest()[:12],
            "source": "gdelt",
            "timestamp": a.get("seendate"),  # e.g. 20261002T153000Z (UTC)
            "ticker": ticker,
            "text": title,
            "url": url,
        })
    return rows


def build_jobs():
    """One job per query: (ticker, GDELT query, headline filter)."""
    universe = pd.read_csv(UNIVERSE_CSV)
    jobs = []
    for r in universe.itertuples():
        headline_terms = r.aliases.split("|") + [r.company_name.split()[0]]
        jobs.append((r.ticker, build_company_query(r.aliases, r.company_name),
                     build_headline_filter(headline_terms)))
    for query, terms in MARKET_QUERIES:
        jobs.append(("MARKET", query, build_headline_filter(terms)))
    return jobs


def run_jobs(jobs, timespan):
    """Run the jobs one by one. Returns (rows, jobs that failed)."""
    rows, failed = [], []
    for i, (ticker, query, headline_filter) in enumerate(jobs):
        if i > 0:
            time.sleep(REQUEST_GAP_SECONDS)
        try:
            articles = fetch_query(query, timespan)
            new_rows = articles_to_rows(articles, ticker, headline_filter)
            rows.extend(new_rows)
            print(f"  {ticker:<7} kept {len(new_rows):>3} of {len(articles):>3} articles")
        except RateLimited:
            print(f"  {ticker:<7} rate limited by GDELT")
            failed.append((ticker, query, headline_filter))
        except GdeltError as e:
            print(f"  {ticker:<7} GDELT ERROR: {e}")
        except requests.RequestException as e:
            print(f"  {ticker:<7} NETWORK ERROR: {e}")
            failed.append((ticker, query, headline_filter))
    return rows, failed


def fetch_all(timespan="7d"):
    """Fetch news for every stock plus the market-wide queries, retrying failures once."""
    rows, failed = run_jobs(build_jobs(), timespan)
    if failed:
        print(f"\nRetrying {len(failed)} queries after a {RETRY_PASS_PAUSE_SECONDS}s pause...")
        time.sleep(RETRY_PASS_PAUSE_SECONDS)
        more_rows, still_failed = run_jobs(failed, timespan)
        rows.extend(more_rows)
        if still_failed:
            names = ", ".join(t for t, _, _ in still_failed)
            print(f"Still failing: {names}. Run the script again later; saved articles are kept.")

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"])


def load_gdelt(path=CACHE_FILE):
    """Load cached GDELT articles (used by demo mode, no internet needed)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[COLUMNS]


def update_cache(new_df, path=CACHE_FILE):
    """Merge newly fetched articles into the cache file, dropping duplicates."""
    old_df = load_gdelt(path)
    frames = [f for f in (old_df, new_df) if len(f)]
    if not frames:
        return old_df, 0
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="id").sort_values("timestamp").reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(path, orient="records", lines=True, date_format="iso")
    return merged, len(merged) - len(old_df)


if __name__ == "__main__":
    n_jobs = len(build_jobs())
    print(f"Fetching {n_jobs} queries from GDELT, one every {REQUEST_GAP_SECONDS}s "
          f"(about {n_jobs * REQUEST_GAP_SECONDS // 60 + 1} min)...")
    fresh = fetch_all()
    cached, n_added = update_cache(fresh)

    print(f"\nNew articles added to cache: {n_added}")
    print(f"Total articles in {CACHE_FILE.name}: {len(cached)}")
    if len(cached):
        print(f"Date range: {cached['timestamp'].min()} -> {cached['timestamp'].max()}")
        print("\nArticles per ticker:")
        print(cached["ticker"].value_counts().to_string())
        print("\nLatest 5 headlines:")
        for record in cached.tail(5).to_dict("records"):
            print(f"  [{record['ticker']}] {record['text']}")