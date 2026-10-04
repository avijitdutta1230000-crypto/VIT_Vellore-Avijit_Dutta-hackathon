"""Impact score (1-10): how much could this piece of news move the market?

    raw    = event_weight x confidence_factor x sentiment_strength x attention
    impact = 1 + 9 x raw   (rounded, kept between 1 and 10)

- event_weight:       how severe this TYPE of event usually is (credit events and
                      geopolitical shocks > earnings > product news > chatter)
- confidence_factor:  0.5 - 1.0, how sure the event classifier is
- sentiment_strength: 0.4 - 1.0, strongly positive/negative text matters more
                      (but big neutral news, e.g. an M&A announcement, still counts)
- attention:          0.5 - 1.25, how unusual today's volume of texts about this
                      stock is versus its own recent normal (an "abnormal attention"
                      spike). Uses only PAST days, so there is no look-ahead bias.

validate() then checks the score against real price moves: do high-impact days
see bigger moves than low-impact days?
"""
import numpy as np
import pandas as pd

from src.ingestion.common import ROOT

PRICES_CSV = ROOT / "data" / "raw" / "stock_yfinance_data.csv"

EVENT_WEIGHTS = {
    "Credit Event": 1.00,
    "Geopolitical": 0.90,
    "Merger/Acquisition": 0.85,
    "Macroeconomic": 0.80,
    "Earnings": 0.75,
    "Legal/Regulatory": 0.75,
    "Analyst Rating": 0.60,
    "Product/Company News": 0.50,
    "Other": 0.20,
}
ATTENTION_WINDOW_DAYS = 30
MARKET_CLOSE_HOUR_ET = 16


def add_signal_date(df):
    """The trading day a text can first affect: its New York date, or the next
    day if it was posted after the 4 pm close."""
    et = df["timestamp"].dt.tz_convert("America/New_York")
    after_close = et.dt.hour >= MARKET_CLOSE_HOUR_ET
    df["signal_date"] = (et.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")).dt.date
    return df


def add_attention(df):
    """attention_ratio = today's text count / median daily count over the past 30 active days."""
    daily = df.groupby(["source", "ticker", "signal_date"]).size().rename("n_today").reset_index()
    daily = daily.sort_values(["source", "ticker", "signal_date"])
    daily["baseline"] = (
        daily.groupby(["source", "ticker"])["n_today"]
        .transform(lambda s: s.shift(1).rolling(ATTENTION_WINDOW_DAYS, min_periods=1).median())
    )
    daily["attention_ratio"] = (daily["n_today"] / daily["baseline"]).fillna(1.0).round(3)
    return df.merge(daily[["source", "ticker", "signal_date", "attention_ratio"]],
                    on=["source", "ticker", "signal_date"], how="left")


def add_impact(df):
    """Add signal_date, attention_ratio and impact (1-10) to a DataFrame that already
    has timestamp, source, ticker, sentiment, event_type and event_confidence."""
    df = add_attention(add_signal_date(df.copy()))
    weight = df["event_type"].map(EVENT_WEIGHTS).fillna(EVENT_WEIGHTS["Other"])
    weight = weight.where(df["topic"] != "Dividend", 0.40)  # routine dividend news rarely moves prices
    confidence = 0.5 + 0.5 * df["event_confidence"].fillna(0)
    strength = 0.4 + 0.6 * df["sentiment"].abs().fillna(0)
    attention = (0.5 + 0.25 * np.log2(1 + df["attention_ratio"])).clip(0.5, 1.25)
    raw = (weight * confidence * strength * attention).clip(0, 1)
    df["impact"] = (1 + 9 * raw).round().clip(1, 10).astype(int)
    return df


def load_prices(path=PRICES_CSV):
    """Daily closes from the Kaggle file, with same-day and next-day returns per ticker."""
    prices = pd.read_csv(path)
    prices = prices.rename(columns={"Stock Name": "ticker", "Date": "date", "Close": "close"})
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["ticker", "date"])
    prices["ret"] = prices.groupby("ticker")["close"].pct_change()
    prices["next_ret"] = prices.groupby("ticker")["ret"].shift(-1)
    return prices[["ticker", "date", "close", "ret", "next_ret"]]


def to_trading_day(signal_dates, trading_dates):
    """Move each date forward to the first trading day on or after it (weekends -> Monday)."""
    trading = np.array(sorted(set(trading_dates)))
    idx = np.searchsorted(trading, np.array(list(signal_dates)), side="left")
    idx = np.clip(idx, 0, len(trading) - 1)
    return trading[idx]


def spearman(a, b):
    """Rank correlation (robust to outliers), computed with pandas only."""
    return round(float(a.rank().corr(b.rank())), 3)


def validate(signals, prices=None):
    """Compare each stock-day's highest impact score with the actual price move.

    Moves are measured relative to each stock's own typical move, so volatile
    stocks like TSLA don't dominate ("1.5x" = 50% bigger than that stock's usual day)."""
    if prices is None:
        prices = load_prices()
    sig = signals[signals["ticker"].isin(prices["ticker"].unique())].copy()
    if sig.empty:
        return None

    sig["trade_date"] = to_trading_day(sig["signal_date"], prices["date"])
    days = sig.groupby(["ticker", "trade_date"]).agg(
        max_impact=("impact", "max"), n_texts=("impact", "size")).reset_index()
    days = days.merge(prices, left_on=["ticker", "trade_date"], right_on=["ticker", "date"], how="inner")

    typical = prices.groupby("ticker")["ret"].apply(lambda r: r.abs().mean())
    days["move_today"] = days["ret"].abs() / days["ticker"].map(typical)
    days["move_next"] = days["next_ret"].abs() / days["ticker"].map(typical)
    days = days.dropna(subset=["move_today", "move_next"])

    days["impact_bucket"] = pd.cut(days["max_impact"], bins=[0, 3, 6, 10],
                                   labels=["low (1-3)", "medium (4-6)", "high (7-10)"])
    buckets = days.groupby("impact_bucket", observed=False).agg(
        stock_days=("move_today", "size"),
        move_same_day=("move_today", "mean"),
        move_next_day=("move_next", "mean")).round(2)

    return {
        "stock_days": int(len(days)),
        "spearman_impact_vs_same_day_move": spearman(days["max_impact"], days["move_today"]),
        "spearman_impact_vs_next_day_move": spearman(days["max_impact"], days["move_next"]),
        "spearman_text_count_vs_same_day_move": spearman(days["n_texts"], days["move_today"]),
        "by_impact_bucket": buckets.reset_index().astype({"impact_bucket": str}).to_dict("records"),
    }