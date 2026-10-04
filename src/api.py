"""Signal API: serves the Risk Engine's structured signals to downstream modules.

Module A subscribes to daily sentiment (/daily), Module B to high-impact events
(/signals/high-impact). Anything else can query individual signals (/signals).

Run from the repo root:   uvicorn src.api:app --port 8000
Interactive docs:          http://localhost:8000/docs
"""
import json
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.ingestion.common import ROOT

SIGNALS_FILE = ROOT / "data" / "signals.jsonl"
DAILY_FILE = ROOT / "data" / "signals_daily.csv"

app = FastAPI(
    title="RiskPulse Signal API",
    description="Structured financial risk signals (sentiment, event type, impact) "
                "extracted from tweets and news by the AI/NLP Risk Engine.",
    version="1.0",
)


@lru_cache(maxsize=1)
def load_signals():
    if not SIGNALS_FILE.exists():
        raise HTTPException(503, "No signals yet. Run `python -m src.pipeline` first.")
    return pd.read_json(SIGNALS_FILE, lines=True, dtype=False, convert_dates=False)


@lru_cache(maxsize=1)
def load_daily():
    if not DAILY_FILE.exists():
        raise HTTPException(503, "No daily summary yet. Run `python -m src.pipeline` first.")
    return pd.read_csv(DAILY_FILE, dtype={"signal_date": str})


def records(df):
    """DataFrame -> plain JSON-ready list (NaN becomes null)."""
    return json.loads(df.to_json(orient="records"))


def filter_signals(ticker=None, source=None, event_type=None, min_impact=1, start=None, end=None):
    df = load_signals()
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    if source:
        df = df[df["source"] == source.lower()]
    if event_type:
        df = df[df["event_type"].str.lower() == event_type.lower()]
    if start:
        df = df[df["signal_date"] >= start]
    if end:
        df = df[df["signal_date"] <= end]
    return df[df["impact"] >= min_impact]


@app.get("/health", summary="API status and what data is loaded")
def health():
    s = load_signals()
    return {
        "status": "ok",
        "signals": int(len(s)),
        "sources": s["source"].value_counts().to_dict(),
        "tickers": sorted(s["ticker"].unique().tolist()),
        "first_signal": s["timestamp"].min(),
        "latest_signal": s["timestamp"].max(),
    }


@app.get("/signals", summary="Query individual signals (newest first)")
def get_signals(
    ticker: Optional[str] = Query(None, description="e.g. AAPL, or MARKET for market-wide news"),
    source: Optional[str] = Query(None, description="tweets or gdelt"),
    event_type: Optional[str] = Query(None, description="e.g. Credit Event, Geopolitical, Earnings"),
    min_impact: int = Query(1, ge=1, le=10),
    start: Optional[str] = Query(None, description="first signal date, YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="last signal date, YYYY-MM-DD"),
    limit: int = Query(100, ge=1, le=5000),
):
    df = filter_signals(ticker, source, event_type, min_impact, start, end)
    page = df.sort_values("timestamp", ascending=False).head(limit)
    return {"total_matching": int(len(df)), "returned": int(len(page)), "signals": records(page)}


@app.get("/signals/latest", summary="Most recent signals across all stocks")
def latest_signals(limit: int = Query(20, ge=1, le=500)):
    page = load_signals().sort_values("timestamp", ascending=False).head(limit)
    return {"returned": int(len(page)), "signals": records(page)}


@app.get("/signals/high-impact", summary="High-impact events (what Module B subscribes to)")
def high_impact(
    min_impact: int = Query(7, ge=1, le=10),
    event_type: Optional[str] = Query(None, description="e.g. Geopolitical"),
    ticker: Optional[str] = None,
    limit: int = Query(100, ge=1, le=5000),
):
    df = filter_signals(ticker=ticker, event_type=event_type, min_impact=min_impact)
    page = df.sort_values(["impact", "timestamp"], ascending=False).head(limit)
    return {"total_matching": int(len(df)), "returned": int(len(page)), "signals": records(page)}


@app.get("/daily", summary="Daily sentiment per stock (what Module A subscribes to)")
def daily(
    ticker: Optional[str] = None,
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(1000, ge=1, le=10000),
):
    df = load_daily()
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    if start:
        df = df[df["signal_date"] >= start]
    if end:
        df = df[df["signal_date"] <= end]
    page = df.sort_values(["signal_date", "ticker"], ascending=False).head(limit)
    return {"total_matching": int(len(df)), "returned": int(len(page)), "days": records(page)}


@app.post("/reload", summary="Reload the files after re-running the pipeline")
def reload():
    load_signals.cache_clear()
    load_daily.cache_clear()
    return {"status": "reloaded", "signals": int(len(load_signals()))}


@app.get("/", include_in_schema=False)
def root():
    """Opening the bare address sends you to the interactive docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/docs")