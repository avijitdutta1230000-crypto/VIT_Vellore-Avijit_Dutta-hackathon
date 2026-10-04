"""Module A: tactical index rebalancer driven by the Risk Engine's sentiment.

Every trading day at the close:
  1. take each stock's impact-weighted sentiment for that day (0 if no news),
     trusting days with only a few texts less (one tweet is weak evidence),
  2. smooth it over time (exponential average, half-life 3 days), so one tweet
     can't whipsaw the index and the signal fades back to neutral when news stops,
  3. tilt weights:  weight = (1/N) x (1 + sensitivity x smoothed sentiment),
     kept between 2% and 15% per stock and summing to 100%,
  4. hold those weights over the NEXT day (no look-ahead: today's weights only
     use news published before today's close).

Benchmark: the same 15 stocks equal-weighted and rebalanced daily, so the only
difference between the two is the sentiment tilt. Trading costs of 10 bps on
every change in weight are charged to the strategy.

Run from the repo root:
    python -m src.modules.rebalancer          # read signals from data/signals_daily.csv
    python -m src.modules.rebalancer --api    # read signals from the running Signal API
"""
import argparse
import json

import numpy as np
import pandas as pd
import requests

from src.ingestion.common import ROOT, load_universe
from src.nlp.impact import to_trading_day

DAILY_FILE = ROOT / "data" / "signals_daily.csv"
PRICES_CSV = ROOT / "data" / "raw" / "stock_yfinance_data.csv"
OUT_DIR = ROOT / "data" / "module_a"
RESULTS_FILE = ROOT / "docs" / "results" / "module_a_backtest.json"
API_URL = "http://localhost:8000/daily"

SENSITIVITY = 3.0       # how strongly sentiment tilts the weights
HALF_LIFE_DAYS = 3      # how fast old news fades
MIN_WEIGHT, MAX_WEIGHT = 0.02, 0.15
COST_BPS = 10           # trading cost per unit of weight changed (10 bps = 0.10%)
SIGNAL_COLUMN = "impact_weighted_sentiment"
SHRINK_TEXTS = 5        # a day with few texts is trusted less: score x n / (n + 5)
TRADING_DAYS = 252


# ---------------------------------------------------------------- inputs

def load_daily_signals(use_api=False):
    """Daily per-stock sentiment from the Risk Engine (file or Signal API)."""
    if use_api:
        resp = requests.get(API_URL, params={"limit": 10000}, timeout=30)
        resp.raise_for_status()
        daily = pd.DataFrame(resp.json()["days"])
        print(f"Subscribed to Signal API: {len(daily):,} stock-days from {API_URL}")
    else:
        daily = pd.read_csv(DAILY_FILE)
    daily["signal_date"] = pd.to_datetime(daily["signal_date"]).dt.date
    return daily


def load_returns(tickers):
    """Daily returns (rows = trading days, columns = tickers) from adjusted closes."""
    prices = pd.read_csv(PRICES_CSV)
    price_col = "Adj Close" if "Adj Close" in prices.columns else "Close"
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
    wide = prices.pivot_table(index="Date", columns="Stock Name", values=price_col).sort_index()
    missing = [t for t in tickers if t not in wide.columns]
    if missing:
        print(f"Warning: no prices for {missing}, they are left out of the index")
    return wide[[t for t in tickers if t in wide.columns]].pct_change()


def sentiment_scores(daily, trading_dates, tickers, half_life=HALF_LIFE_DAYS):
    """Smoothed sentiment per stock per trading day (0 = neutral / no news)."""
    d = daily[daily["ticker"].isin(tickers) & (daily["signal_date"] <= max(trading_dates))].copy()
    d["trade_date"] = to_trading_day(d["signal_date"], trading_dates)
    # Weekend news lands on Monday: combine those days, weighted by how many texts each had.
    d["weighted"] = d[SIGNAL_COLUMN] * d["n_texts"]
    agg = d.groupby(["trade_date", "ticker"])[["weighted", "n_texts"]].sum()
    # Shrink towards neutral when there are only a few texts (one tweet is weak evidence).
    trust = agg["n_texts"] / (agg["n_texts"] + SHRINK_TEXTS)
    raw = (agg["weighted"] / agg["n_texts"] * trust).unstack("ticker")
    raw = raw.reindex(index=trading_dates, columns=tickers).fillna(0.0)
    return raw.ewm(halflife=half_life, adjust=False).mean()


# ---------------------------------------------------------------- strategy

def bounded_weights(raw, lo=MIN_WEIGHT, hi=MAX_WEIGHT):
    """Scale to sum to 1 while keeping every weight between lo and hi."""
    w = np.clip(np.asarray(raw, dtype=float), 1e-9, None)
    lo, hi = min(lo, 1 / len(w)), max(hi, 1 / len(w))  # keep the bounds possible for small indexes
    for _ in range(100):
        w = np.clip(w / w.sum(), lo, hi)
        if abs(w.sum() - 1) < 1e-10:
            break
    return w / w.sum()


def target_weights(scores, sensitivity=SENSITIVITY):
    """Sentiment tilt: weight = (1/N) x (1 + sensitivity x score), then bounded."""
    n = scores.shape[1]
    tilted = (1.0 / n) * (1 + sensitivity * scores.values)
    return pd.DataFrame([bounded_weights(row) for row in tilted], index=scores.index, columns=scores.columns)


def backtest(returns, weights, cost_bps=COST_BPS):
    """Weights set at day t's close earn day t+1's returns."""
    held = weights.shift(1)                                   # yesterday's decision
    turnover = weights.diff().abs().sum(axis=1)               # weight traded at each close
    strategy = (held * returns).sum(axis=1, min_count=1) - (cost_bps / 1e4) * turnover.shift(1)
    baseline = returns.mean(axis=1)                           # equal weight, rebalanced daily
    out = pd.DataFrame({"strategy": strategy, "baseline": baseline, "turnover": turnover}).iloc[2:]
    return out.dropna(subset=["strategy", "baseline"])


def performance(daily_returns):
    """Standard performance numbers for a series of daily returns."""
    value = (1 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    ann = value.iloc[-1] ** (1 / years) - 1
    return {
        "total_return": round(float(value.iloc[-1] - 1), 4),
        "annual_return": round(float(ann), 4),
        "annual_volatility": round(float(vol), 4),
        "sharpe_ratio": round(float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS)), 3),
        "max_drawdown": round(float((value / value.cummax() - 1).min()), 4),
    }


def run(daily, returns, sensitivity=SENSITIVITY, half_life=HALF_LIFE_DAYS):
    scores = sentiment_scores(daily, list(returns.index), list(returns.columns), half_life)
    weights = target_weights(scores, sensitivity)
    return scores, weights, backtest(returns, weights)


def live_weights(daily, tickers, after_date, half_life=HALF_LIFE_DAYS, sensitivity=SENSITIVITY):
    """Today's suggested weights from the newest signals (live GDELT news)."""
    live = daily[(daily["signal_date"] > after_date) & daily["ticker"].isin(tickers)]
    if live.empty:
        return None
    days = pd.date_range(min(live["signal_date"]), max(live["signal_date"])).date
    raw = live.pivot_table(index="signal_date", columns="ticker", values=SIGNAL_COLUMN)
    raw = raw.reindex(index=days, columns=tickers).fillna(0.0)
    scores = raw.ewm(halflife=half_life, adjust=False).mean().iloc[[-1]]
    w = target_weights(scores, sensitivity).iloc[0]
    return pd.DataFrame({"ticker": tickers, "score": scores.iloc[0].round(4).values,
                         "weight": w.round(4).values, "equal_weight": round(1 / len(tickers), 4),
                         "as_of": max(live["signal_date"])})


# ---------------------------------------------------------------- report

def main(use_api=False):
    tickers = load_universe()["ticker"].tolist()
    daily = load_daily_signals(use_api)
    returns = load_returns(tickers)
    tickers = list(returns.columns)

    scores, weights, result = run(daily, returns)
    strat, base = performance(result["strategy"]), performance(result["baseline"])
    active = result["strategy"] - result["baseline"]
    info_ratio = round(float(active.mean() / active.std() * np.sqrt(TRADING_DAYS)), 3)
    avg_turnover = round(float(result["turnover"].mean()), 4)

    start, end = result.index.min(), result.index.max()
    print(f"\nBacktest {start} -> {end}  ({len(result)} trading days, {len(tickers)} stocks)")
    table = pd.DataFrame({"Sentiment-tilted": strat, "Equal-weight": base})
    print(table.to_string())
    print(f"Information ratio (excess return per unit of tracking risk): {info_ratio}")
    print(f"Average daily turnover: {avg_turnover:.2%} of the portfolio (cost {COST_BPS} bps)")

    print("\nRobustness: excess annual return vs equal-weight for other settings")
    grid = {}
    for k in (1.0, 3.0, 5.0):
        row = {}
        for hl in (1, 3, 7):
            r = run(daily, returns, sensitivity=k, half_life=hl)[2]
            row[f"half-life {hl}d"] = round(performance(r["strategy"])["annual_return"] - base["annual_return"], 4)
        grid[f"sensitivity {k:g}"] = row
    grid_df = pd.DataFrame(grid).T
    print(grid_df.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.round(4).to_csv(OUT_DIR / "weights_history.csv", index_label="date")
    curve = (1 + result[["strategy", "baseline"]]).cumprod().round(5)
    curve.to_csv(OUT_DIR / "equity_curve.csv", index_label="date")

    print("\nBiggest weight changes over the backtest (max - min weight):")
    spread = (weights.max() - weights.min()).sort_values(ascending=False).round(3)
    print(spread.head(5).to_string())

    live = live_weights(daily, tickers, after_date=end)
    if live is not None:
        live.to_csv(OUT_DIR / "live_weights.csv", index=False)
        print(f"\nToday's suggested weights from live news (as of {live['as_of'].iloc[0]}):")
        print(live.sort_values("weight", ascending=False)[["ticker", "score", "weight"]].to_string(index=False))

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({
        "period": [str(start), str(end)], "trading_days": int(len(result)), "stocks": tickers,
        "settings": {"sensitivity": SENSITIVITY, "half_life_days": HALF_LIFE_DAYS, "shrink_texts": SHRINK_TEXTS,
                     "min_weight": MIN_WEIGHT, "max_weight": MAX_WEIGHT, "cost_bps": COST_BPS,
                     "signal": SIGNAL_COLUMN},
        "sentiment_tilted": strat, "equal_weight": base,
        "information_ratio": info_ratio, "avg_daily_turnover": avg_turnover,
        "robustness_excess_annual_return": grid_df.to_dict(),
    }, indent=2, default=str))
    print(f"\nSaved weights, equity curve and results to {OUT_DIR} and {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module A: sentiment-driven index rebalancer")
    parser.add_argument("--api", action="store_true", help="read signals from the running Signal API")
    main(use_api=parser.parse_args().api)