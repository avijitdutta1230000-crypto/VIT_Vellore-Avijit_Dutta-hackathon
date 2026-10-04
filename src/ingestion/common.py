"""Shared helpers for all ingestion sources (tweets, GDELT).

Every source returns the same columns, and every source uses the same
entity-linking rule: a text counts for a company only if it actually
mentions that company (by name, alias or cashtag).
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_CSV = ROOT / "data" / "universe.csv"

# Every ingestion source returns these same columns.
COLUMNS = ["id", "source", "timestamp", "ticker", "text", "url"]

# Old or alternative tickers people still use in cashtags.
# Meta traded as $FB until June 2022, which covers most of our tweet period.
EXTRA_CASHTAGS = {"GOOG": ["GOOGL"], "META": ["FB"]}

URL_RE = re.compile(r"https?://\S+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    """Remove links and extra whitespace. Cashtags and hashtags are kept."""
    text = URL_RE.sub("", str(text))
    return SPACE_RE.sub(" ", text).strip()


def load_universe(path=UNIVERSE_CSV):
    """The stocks in our mock index: ticker, company_name, sector, aliases."""
    return pd.read_csv(path)


def build_mention_filter(terms, cashtags=()):
    """Regex that matches any of the terms (as whole words) or cashtags in a text.

    Short terms with capitals (AMD, AWS, P&G, Meta, Fed, CPI) are case-sensitive,
    so "meta" or "fed up" don't count. Longer terms ignore case and allow a
    plural ending ("tariff" also matches "tariffs"). Cashtags ignore case
    ($amzn = $AMZN) but must end there ($TSLAQ is not $TSLA).
    """
    parts = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        escaped = re.escape(term)
        if len(term) <= 4 and any(c.isupper() for c in term):
            parts.append(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
        else:
            parts.append(rf"(?i:(?<![A-Za-z0-9]){escaped}(?:s|es)?(?![A-Za-z0-9]))")
    for tag in cashtags:
        parts.append(rf"(?i:\${re.escape(tag)}(?![A-Za-z0-9]))")
    return re.compile("|".join(parts))


def company_filters(universe=None):
    """{ticker: regex} that detects a mention of each company in the universe."""
    if universe is None:
        universe = load_universe()
    return {
        r.ticker: build_mention_filter(r.aliases.split("|"), [r.ticker] + EXTRA_CASHTAGS.get(r.ticker, []))
        for r in universe.itertuples()
    }