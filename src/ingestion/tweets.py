"""Load the Kaggle stock tweets dataset into the common ingestion schema.

Source: "Stock Tweets for Sentiment Analysis and Prediction" (Kaggle, equinxx).
The tweets are historical (2021-09-30 to 2022-09-29), so they are replayed
in time order to simulate a live social media feed.

Data-quality fix: in the raw file, the same ~4,080 tweets (mostly about Amazon)
are copied under MSFT, PG and AMZN. So a tweet is kept for a stock only if it
actually mentions that company (name, alias or cashtag) - the same
entity-linking rule we use for news headlines.

Run from the repo root:  python -m src.ingestion.tweets
"""
import hashlib
from pathlib import Path

import pandas as pd

from src.ingestion.common import COLUMNS, ROOT, clean_text, company_filters, load_universe

TWEETS_CSV = ROOT / "data" / "raw" / "stock_tweets.csv"


def make_id(ticker, timestamp, text):
    """Stable ID, so the same tweet always gets the same ID across runs."""
    raw = f"{ticker}|{timestamp}|{text}".encode("utf-8")
    return "tw_" + hashlib.sha1(raw).hexdigest()[:12]


def load_tweets(path=TWEETS_CSV, tickers=None, start=None, end=None, verbose=False):
    """Load tweets for the given tickers in the common schema, oldest first.

    start / end are optional date strings like "2022-01-01" (end is exclusive).
    """
    df = pd.read_csv(path)
    n_raw = len(df)

    df = df.rename(columns={"Date": "timestamp", "Tweet": "raw_text", "Stock Name": "ticker"})

    universe = load_universe()
    if tickers is not None:
        universe = universe[universe["ticker"].isin(tickers)]
    filters = company_filters(universe)

    df = df[df["ticker"].isin(filters)].copy()
    n_universe = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")]

    df["text"] = df["raw_text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    # Entity linking: keep the tweet only if it mentions the company it is filed under.
    mentions = [filters[t].search(x) is not None for t, x in zip(df["ticker"], df["text"])]
    n_before_mention = len(df)
    df = df[mentions]
    n_no_mention = n_before_mention - len(df)

    # The same text posted repeatedly about the same stock is usually spam or bots.
    # Sort first so we keep the earliest copy of each duplicate.
    df = df.sort_values("timestamp")
    n_before_dedupe = len(df)
    df = df.drop_duplicates(subset=["ticker", "text"], keep="first")

    df["source"] = "tweets"
    df["url"] = None
    df["id"] = [make_id(t, ts, x) for t, ts, x in zip(df["ticker"], df["timestamp"], df["text"])]
    df = df[COLUMNS].reset_index(drop=True)

    if verbose:
        print(f"Raw tweets in file:                {n_raw:,}")
        print(f"Filed under our universe:          {n_universe:,}")
        print(f"Dropped, company not mentioned:    {n_no_mention:,}")
        print(f"Dropped, duplicate spam:           {n_before_dedupe - len(df):,}")
        print(f"Final tweets:                      {len(df):,}")

    return df


def replay(df):
    """Yield tweets one at a time in time order, like a live feed."""
    for record in df.to_dict("records"):
        yield record


if __name__ == "__main__":
    tweets = load_tweets(verbose=True)
    print(f"Date range: {tweets['timestamp'].min()} -> {tweets['timestamp'].max()}")
    print("\nTweets per ticker:")
    print(tweets["ticker"].value_counts().to_string())
    print("\nFirst 3 records:")
    for i, record in enumerate(replay(tweets)):
        print(record)
        if i == 2:
            break