"""Fetch recent global news from the GDELT DOC 2.0 API into the common schema.

GDELT monitors world news and updates every 15 minutes. The DOC API is free and
needs no API key. It covers roughly the last 3 months, returns at most 250
articles per request, and asks for no more than one request every 5 seconds.

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
REQUEST_GAP_SECONDS = 6  # GDELT asks for at most one request every 5 seconds

# Company queries must also contain a finance word, so "Amazon" the rainforest
# or "Apple" the fruit don't sneak in.
FINANCE_TERMS = "(stock OR shares OR earnings OR investors OR analysts)"

# Market-wide queries. These rows get ticker = "MARKET" and feed the
# Macroeconomic / Geopolitical event classes and the Module B stress test.
MARKET_QUERIES = [
    '("Federal Reserve" OR "interest rates" OR inflation OR recession OR "central bank")',
    '(sanctions OR tariffs OR "trade war" OR invasion OR "military conflict") (markets OR stocks OR investors)',
]

# GDELT rejects very short keywords and some symbols, so aliases like
# "AMD", "AWS" or "P&G" are skipped for GDELT searches.
SAFE_ALIAS = re.compile(r"^[A-Za-z0-9 .\-]{4,}$")


class GdeltError(Exception):
    pass


def build_company_query(aliases, company_name):
    """'Apple|iPhone' -> ("Apple" OR "iPhone") (stock OR shares OR ...)"""
    terms = [a.strip() for a in aliases.split("|") if SAFE_ALIAS.match(a.strip())]
    if not terms:
        # e.g. Procter & Gamble: fall back to the first word of the company name
        terms = [company_name.split()[0]]
    names = " OR ".join(f'"{t}"' for t in terms)
    if len(terms) > 1:
        names = f"({names})"  # GDELT only allows brackets around OR'd terms
    return f"{names} {FINANCE_TERMS}"


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

    for attempt in range(2):
        resp = requests.get(API_URL, params=params, headers=headers, timeout=60)
        text = resp.text.strip()

        if "limit requests" in text.lower() or resp.status_code == 429:
            time.sleep(15)  # throttled: wait, then try once more
            continue
        if not text:
            return []
        try:
            data = json.loads(text, strict=False)  # strict=False tolerates odd characters in titles
        except json.JSONDecodeError:
            # GDELT replies with plain text (not JSON) when it rejects a query
            raise GdeltError(text[:200])
        return data.get("articles", [])

    raise GdeltError("rate limited twice in a row, try again in a minute")


def articles_to_rows(articles, ticker):
    """Convert GDELT articles into rows of the common schema."""
    rows = []
    for a in articles:
        title = clean_text(a.get("title") or "")
        url = a.get("url")
        if not title or not url:
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


def fetch_all(timespan="7d"):
    """Fetch news for every stock in the universe plus the market-wide queries."""
    universe = pd.read_csv(UNIVERSE_CSV)
    queries = [(r.ticker, build_company_query(r.aliases, r.company_name)) for r in universe.itertuples()]
    queries += [("MARKET", q) for q in MARKET_QUERIES]

    rows = []
    for ticker, query in queries:
        try:
            new_rows = articles_to_rows(fetch_query(query, timespan), ticker)
            rows.extend(new_rows)
            print(f"  {ticker:<7} {len(new_rows):>3} articles")
        except GdeltError as e:
            print(f"  {ticker:<7} GDELT ERROR: {e}")
        except requests.RequestException as e:
            print(f"  {ticker:<7} NETWORK ERROR: {e}")
        time.sleep(REQUEST_GAP_SECONDS)

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
    n_queries = len(pd.read_csv(UNIVERSE_CSV)) + len(MARKET_QUERIES)
    print(f"Fetching {n_queries} queries from GDELT (about {n_queries * REQUEST_GAP_SECONDS // 60 + 1} min, "
          f"GDELT allows one request every 5 seconds)...")
    fresh = fetch_all()
    cached, n_added = update_cache(fresh)

    print(f"\nNew articles added to cache: {n_added}")
    print(f"Total articles in {CACHE_FILE.name}: {len(cached)}")
    if len(cached):
        print(f"Date range: {cached['timestamp'].min()} -> {cached['timestamp'].max()}")
        print("\nArticles per ticker:")
        print(cached["ticker"].value_counts().to_string())
        print("\nLatest 3 articles:")
        for record in cached.tail(3).to_dict("records"):
            print(record)