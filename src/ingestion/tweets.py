"""Load the Kaggle stock tweets dataset into the common ingestion schema.

Source: "Stock Tweets for Sentiment Analysis and Prediction" (Kaggle, equinxx).
The tweets are historical (2021-09-30 to 2022-09-29), so they are replayed
in time order to simulate a live social media feed.

Run from the repo root:  python -m src.ingestion.tweets
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TWEETS_CSV = ROOT / "data" / "raw" / "stock_tweets.csv"
UNIVERSE_CSV = ROOT / "data" / "universe.csv"

# Every ingestion source (tweets, NewsAPI, GDELT) returns these same columns.
COLUMNS = ["id", "source", "timestamp", "ticker", "text", "url"]

URL_RE = re.compile(r"https?://\S+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    """Remove links and extra whitespace. Cashtags and hashtags are kept."""
    text = URL_RE.sub("", str(text))
    return SPACE_RE.sub(" ", text).strip()


def make_id(ticker, timestamp, text):
    """Stable ID, so the same tweet always gets the same ID across runs."""
    raw = f"{ticker}|{timestamp}|{text}".encode("utf-8")
    return "tw_" + hashlib.sha1(raw).hexdigest()[:12]


def load_universe(path=UNIVERSE_CSV):
    """Return the list of tickers in our mock index."""
    return pd.read_csv(path)["ticker"].tolist()


def load_tweets(path=TWEETS_CSV, tickers=None, start=None, end=None, verbose=False):
    """Load tweets for the given tickers in the common schema, oldest first.

    start / end are optional date strings like "2022-01-01" (end is exclusive).
    """
    df = pd.read_csv(path)
    n_raw = len(df)

    df = df.rename(columns={"Date": "timestamp", "Tweet": "raw_text", "Stock Name": "ticker"})

    if tickers is None:
        tickers = load_universe()
    df = df[df["ticker"].isin(tickers)].copy()
    n_universe = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")]

    df["text"] = df["raw_text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

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
        print(f"Raw tweets in file:          {n_raw:,}")
        print(f"In our stock universe:        {n_universe:,}")
        print(f"Duplicates removed:          {n_before_dedupe - len(df):,}")
        print(f"Final tweets:                {len(df):,}")

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