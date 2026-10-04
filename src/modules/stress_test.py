"""Module B: event-driven stress testing of a synthetic wholesale banking portfolio.

Subscribes to the Risk Engine's event type + impact score. When a high-impact event
arrives (impact >= 7), it picks a shock scenario and revalues the portfolio:

  Market-wide events (ticker MARKET, or Geopolitical / Macroeconomic news)
      -> a market scenario: equities, interest rates, credit spreads, USD, default risk
         (at half strength when the news is about one company, e.g. "Netflix hit by strong dollar")
  Company events with negative sentiment (e.g. a credit or legal shock at Disney)
      -> an issuer scenario: only positions linked to that company are hit

Shocks are calibrated for an impact-8 event and scaled by impact (impact 10 = 1.25x).

Valuation is deliberately simple (first-order sensitivities, as in a quick risk check):
  equity            dV = value x beta x equity shock
  bonds             dV = -value x duration x (rate change + spread change)
  loans (floating)  dV = -value x spread duration x spread change - expected-loss increase
  interest swaps    dV = +/- notional x duration x rate change   (+ pay-fixed, - receive-fixed)
  FX forwards       dV = -notional x USD strengthening           (long foreign currency)
  CDS               dV = +/- notional x spread duration x spread change (+ protection bought)

The portfolio is fully synthetic (fictional book, no real client data).

Run from the repo root:
    python -m src.modules.stress_test          # read signals from data/signals.jsonl
    python -m src.modules.stress_test --api    # subscribe via the running Signal API
"""
import argparse
import json

import pandas as pd
import requests

from src.ingestion.common import ROOT

SIGNALS_FILE = ROOT / "data" / "signals.jsonl"
OUT_DIR = ROOT / "data" / "module_b"
PORTFOLIO_FILE = OUT_DIR / "portfolio.csv"
RESULTS_FILE = ROOT / "docs" / "results" / "module_b_summary.json"
API_URL = "http://localhost:8000/signals/high-impact"

IMPACT_THRESHOLD = 7
CALIBRATION_IMPACT = 8          # shocks below are for an impact-8 event
ISSUER_SENTIMENT_TRIGGER = -0.2  # company news must be clearly negative to count as stress
SPILLOVER_SEVERITY = 0.5        # a company story with a macro/geopolitical theme is weaker
                                # evidence of a market-wide shock than market-wide news itself
LOAN_LGD = 0.45                 # loss given default for loans

# One-year default probability by rating (simplified, rating-agency style).
PD_BY_RATING = {"AAA": 0.0001, "AA": 0.0002, "A": 0.0006, "BBB": 0.002, "BB": 0.01, "B": 0.04, "CCC": 0.15}
INVESTMENT_GRADE = {"AAA", "AA", "A", "BBB"}

# Market scenarios (impact 8). rates and spreads in decimal (0.02 = +200 bp).
MARKET_SCENARIOS = {
    "Geopolitical": {"equity": -0.10, "rates": -0.0025, "ig_spread": 0.0075, "hy_spread": 0.0200,
                     "usd": 0.05, "pd_multiplier": 1.5,
                     "story": "Risk-off: stocks fall, investors flee to safe government bonds, USD rises"},
    "Macroeconomic": {"equity": -0.05, "rates": 0.0200, "ig_spread": 0.0025, "hy_spread": 0.0075,
                      "usd": 0.03, "pd_multiplier": 1.25,
                      "story": "Rate shock: yields jump 2%, bond prices fall, stocks dip"},
    "Credit Event": {"equity": -0.07, "rates": -0.0050, "ig_spread": 0.0100, "hy_spread": 0.0300,
                     "usd": 0.02, "pd_multiplier": 2.0,
                     "story": "Credit crunch: spreads blow out, default risk doubles"},
}
# Issuer scenarios (impact 8): only positions on that company are hit.
ISSUER_SCENARIOS = {
    "credit": {"equity": -0.20, "spread": 0.0300, "pd_multiplier": 4.0,
               "story": "Company credit shock: its bonds and loans reprice, default risk 4x"},
    "default": {"equity": -0.15, "spread": 0.0150, "pd_multiplier": 2.5,
                "story": "Company-specific bad news: its stock and debt reprice"},
}
CREDIT_LIKE = {"Credit Event", "Legal/Regulatory"}


# ---------------------------------------------------------------- portfolio

def build_portfolio():
    """A fictional wholesale banking book (~$1bn). Issuers overlap with the
    engine's stock universe, so company news can hit specific positions."""
    rows = [
        # id, asset_class, issuer, rating, value, notional, duration, beta, direction, currency
        ("EQ-01", "Equity", "AAPL", "AA", 15e6, 0, 0, 1.2, 1, "USD"),
        ("EQ-02", "Equity", "MSFT", "AAA", 15e6, 0, 0, 1.1, 1, "USD"),
        ("EQ-03", "Equity", "TSLA", "BBB", 8e6, 0, 0, 2.0, 1, "USD"),
        ("EQ-04", "Equity", "DIS", "A", 10e6, 0, 0, 1.1, 1, "USD"),
        ("EQ-05", "Equity", "BA", "BBB", 8e6, 0, 0, 1.4, 1, "USD"),
        ("EQ-06", "Equity", "NFLX", "BBB", 6e6, 0, 0, 1.3, 1, "USD"),
        ("CB-01", "Corporate Bond", "AAPL", "AA", 40e6, 0, 6.5, 0, 1, "USD"),
        ("CB-02", "Corporate Bond", "MSFT", "AAA", 40e6, 0, 8.0, 0, 1, "USD"),
        ("CB-03", "Corporate Bond", "DIS", "A", 35e6, 0, 7.0, 0, 1, "USD"),
        ("CB-04", "Corporate Bond", "BA", "BBB", 30e6, 0, 5.5, 0, 1, "USD"),
        ("CB-05", "Corporate Bond", "INTC", "A", 25e6, 0, 6.0, 0, 1, "USD"),
        ("CB-06", "Corporate Bond", "PYPL", "A", 20e6, 0, 4.5, 0, 1, "USD"),
        ("CB-07", "Corporate Bond", "Meridian Chemicals", "B", 15e6, 0, 4.0, 0, 1, "USD"),
        ("GB-01", "Government Bond", "US Treasury 2Y", "AAA", 120e6, 0, 1.9, 0, 1, "USD"),
        ("GB-02", "Government Bond", "US Treasury 10Y", "AAA", 150e6, 0, 8.5, 0, 1, "USD"),
        ("GB-03", "Government Bond", "US Treasury 30Y", "AAA", 50e6, 0, 18.0, 0, 1, "USD"),
        ("LN-01", "Corporate Loan", "BA", "BBB", 60e6, 0, 2.0, 0, 1, "USD"),
        ("LN-02", "Corporate Loan", "DIS", "A", 50e6, 0, 2.0, 0, 1, "USD"),
        ("LN-03", "Corporate Loan", "Harbor Freight Lines", "BBB", 70e6, 0, 2.5, 0, 1, "USD"),
        ("LN-04", "Corporate Loan", "Northgate Retail", "BB", 55e6, 0, 2.0, 0, 1, "USD"),
        ("LN-05", "Corporate Loan", "Meridian Chemicals", "B", 40e6, 0, 1.5, 0, 1, "USD"),
        ("IRS-01", "Interest Rate Swap", "Pay fixed 5Y (hedge)", "AA", 0, 100e6, 4.5, 0, 1, "USD"),
        ("IRS-02", "Interest Rate Swap", "Receive fixed 10Y", "AA", 0, 60e6, 8.5, 0, -1, "USD"),
        ("FX-01", "FX Forward", "Long EUR / short USD", "A", 0, 40e6, 0, 0, 1, "EUR"),
        ("FX-02", "FX Forward", "Long JPY / short USD", "A", 0, 25e6, 0, 0, 1, "JPY"),
        ("CDS-01", "Credit Default Swap", "BA", "BBB", 0, 30e6, 4.5, 0, 1, "USD"),          # protection bought
        ("CDS-02", "Credit Default Swap", "CDX IG index", "A", 0, 80e6, 4.5, 0, -1, "USD"),  # protection sold
    ]
    cols = ["position_id", "asset_class", "issuer", "rating", "value", "notional", "duration",
            "beta", "direction", "currency"]
    return pd.DataFrame(rows, columns=cols)


def load_portfolio():
    if not PORTFOLIO_FILE.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        build_portfolio().to_csv(PORTFOLIO_FILE, index=False)
    return pd.read_csv(PORTFOLIO_FILE)


# ---------------------------------------------------------------- valuation

def revalue(portfolio, shocks):
    """Value change of every position under one set of shocks.

    shocks: equity, rates, ig_spread, hy_spread, usd, pd_multiplier, and optionally
    issuer + issuer_equity / issuer_spread / issuer_pd_multiplier for company events."""
    p = portfolio.copy()
    issuer = shocks.get("issuer")
    on_issuer = p["issuer"] == issuer if issuer else pd.Series(False, index=p.index)

    ig = p["rating"].isin(INVESTMENT_GRADE)
    spread = ig * shocks.get("ig_spread", 0) + (~ig) * shocks.get("hy_spread", 0)
    spread = spread + on_issuer * shocks.get("issuer_spread", 0)
    equity = shocks.get("equity", 0) + on_issuer * shocks.get("issuer_equity", 0)
    pd_mult = pd.Series(shocks.get("pd_multiplier", 1.0), index=p.index)
    pd_mult = pd_mult.where(~on_issuer, pd_mult * shocks.get("issuer_pd_multiplier", 1.0))
    rates, usd = shocks.get("rates", 0), shocks.get("usd", 0)
    base_pd = p["rating"].map(PD_BY_RATING).fillna(0.01)

    a = p["asset_class"]
    dv = pd.Series(0.0, index=p.index)
    dv[a == "Equity"] = (p["value"] * p["beta"] * equity)[a == "Equity"]
    dv[a == "Corporate Bond"] = (-p["value"] * p["duration"] * (rates + spread))[a == "Corporate Bond"]
    dv[a == "Government Bond"] = (-p["value"] * p["duration"] * rates)[a == "Government Bond"]
    loan_el = p["value"] * LOAN_LGD * base_pd * (pd_mult - 1)
    dv[a == "Corporate Loan"] = (-p["value"] * p["duration"] * spread - loan_el)[a == "Corporate Loan"]
    dv[a == "Interest Rate Swap"] = (p["direction"] * p["notional"] * p["duration"] * rates)[a == "Interest Rate Swap"]
    dv[a == "FX Forward"] = (-p["direction"] * p["notional"] * usd)[a == "FX Forward"]
    dv[a == "Credit Default Swap"] = (p["direction"] * p["notional"] * p["duration"] * spread)[a == "Credit Default Swap"]

    p["value_change"] = dv.round(0)
    return p


def scenario_for(signal):
    """Pick and scale the shocks for one triggering signal. Returns None if it isn't a stress event."""
    severity = signal["impact"] / CALIBRATION_IMPACT
    event, ticker = signal["event_type"], signal["ticker"]

    if ticker == "MARKET" or event in ("Geopolitical", "Macroeconomic"):
        base = MARKET_SCENARIOS.get(event) or (MARKET_SCENARIOS["Credit Event"] if event in CREDIT_LIKE else None)
        if base is None:
            return None
        name = f"Market: {event}"
        if ticker != "MARKET":
            severity *= SPILLOVER_SEVERITY
        shocks = {k: v * severity for k, v in base.items() if k not in ("pd_multiplier", "story")}
        shocks["pd_multiplier"] = 1 + (base["pd_multiplier"] - 1) * severity
        return name, base["story"], shocks

    if signal["sentiment"] > ISSUER_SENTIMENT_TRIGGER:
        return None  # positive or neutral company news is not a stress event
    base = ISSUER_SCENARIOS["credit" if event in CREDIT_LIKE else "default"]
    shocks = {"issuer": ticker,
              "issuer_equity": base["equity"] * severity,
              "issuer_spread": base["spread"] * severity,
              "issuer_pd_multiplier": 1 + (base["pd_multiplier"] - 1) * severity}
    return f"Issuer: {ticker} ({event})", base["story"], shocks


# ---------------------------------------------------------------- engine subscription

def load_triggers(use_api=False, threshold=IMPACT_THRESHOLD):
    """High-impact signals from the Risk Engine (file or Signal API)."""
    if use_api:
        resp = requests.get(API_URL, params={"min_impact": threshold, "limit": 5000}, timeout=30)
        resp.raise_for_status()
        sig = pd.DataFrame(resp.json()["signals"])
        print(f"Subscribed to Signal API: {len(sig)} signals with impact >= {threshold}")
    else:
        sig = pd.read_json(SIGNALS_FILE, lines=True, dtype=False, convert_dates=False)
        sig = sig[sig["impact"] >= threshold]
    return sig


def run_stress_tests(signals, portfolio):
    """One stress test per (day, scenario). Market-wide news is preferred over company
    stories, then the highest impact, to drive each test. Company events where the
    portfolio has no exposure are counted but not stress-tested."""
    value_before = float(portfolio["value"].sum())
    held_issuers = set(portfolio["issuer"])
    tests, breakdowns, no_exposure = {}, [], 0
    ordered = signals.assign(is_market=signals["ticker"] == "MARKET").sort_values(
        ["is_market", "impact", "timestamp"], ascending=False)
    for s in ordered.to_dict("records"):
        picked = scenario_for(s)
        if picked is None:
            continue
        name, story, shocks = picked
        if "issuer" in shocks and shocks["issuer"] not in held_issuers:
            no_exposure += 1
            continue
        key = (s["signal_date"], name)
        if key in tests:
            tests[key]["n_signals"] += 1
            continue
        revalued = revalue(portfolio, shocks)
        change = float(revalued["value_change"].sum())
        test_id = f"ST-{len(tests) + 1:03d}"
        by_class = revalued.groupby("asset_class")["value_change"].sum()
        worst = revalued.nsmallest(1, "value_change").iloc[0]
        tests[key] = {
            "test_id": test_id, "signal_date": s["signal_date"], "scenario": name, "story": story,
            "trigger_ticker": s["ticker"], "event_type": s["event_type"], "impact": int(s["impact"]),
            "sentiment": s["sentiment"], "source": s["source"], "headline": s["text"], "url": s.get("url"),
            "value_before": round(value_before), "value_after": round(value_before + change),
            "pnl": round(change), "pnl_pct": round(change / value_before, 5),
            "worst_position": f"{worst['position_id']} {worst['asset_class']} ({worst['issuer']})",
            "worst_position_pnl": round(float(worst["value_change"])), "n_signals": 1,
        }
        breakdowns.append(by_class.rename("pnl").reset_index().assign(test_id=test_id))
        breakdowns.append(revalued[["position_id", "asset_class", "issuer", "value_change"]]
                          .rename(columns={"value_change": "pnl"}).assign(test_id=test_id, level="position"))
    events = pd.DataFrame(tests.values())
    events.attrs["no_exposure"] = no_exposure
    if events.empty:
        return events, pd.DataFrame()
    detail = pd.concat(breakdowns, ignore_index=True)
    detail["level"] = detail["level"].fillna("asset_class")
    return events.sort_values("pnl").reset_index(drop=True), detail


def what_if_table(portfolio):
    """Portfolio P&L for each market scenario at impact 8 and 10 (no news needed)."""
    rows = []
    for event in MARKET_SCENARIOS:
        for impact in (8, 10):
            fake = {"impact": impact, "event_type": event, "ticker": "MARKET", "sentiment": -1}
            _, _, shocks = scenario_for(fake)
            change = float(revalue(portfolio, shocks)["value_change"].sum())
            rows.append({"scenario": event, "impact": impact, "pnl": round(change),
                         "pnl_pct": round(change / portfolio["value"].sum(), 5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report

def main(use_api=False):
    portfolio = load_portfolio()
    total = portfolio["value"].sum()
    print(f"Synthetic portfolio: {len(portfolio)} positions, ${total / 1e6:,.0f}m invested "
          f"(+ ${portfolio['notional'].sum() / 1e6:,.0f}m derivative notional)")
    print((portfolio.groupby("asset_class")["value"].sum() / 1e6).round(0).to_string())

    signals = load_triggers(use_api)
    events, detail = run_stress_tests(signals, portfolio)
    what_if = what_if_table(portfolio)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_DIR / "stress_events.csv", index=False)
    detail.to_csv(OUT_DIR / "stress_detail.csv", index=False)
    what_if.to_csv(OUT_DIR / "what_if.csv", index=False)

    print(f"\nWhat-if: portfolio P&L if each market scenario hit today")
    show = what_if.assign(pnl=lambda d: (d["pnl"] / 1e6).round(1), pnl_pct=lambda d: (d["pnl_pct"] * 100).round(2))
    print(show.rename(columns={"pnl": "P&L $m", "pnl_pct": "P&L %"}).to_string(index=False))

    if events.empty:
        print(f"\nNo signals with impact >= {IMPACT_THRESHOLD} triggered a stress test.")
        return
    print(f"\n{len(events)} stress tests triggered by high-impact events "
          f"({(events['source'] == 'gdelt').sum()} from live news). "
          f"{events.attrs.get('no_exposure', 0)} company events skipped: no exposure in the portfolio.")
    print(events["scenario"].str.split(":").str[0].value_counts().to_string())
    print("\nWorst 5 stress tests:")
    for r in events.head(5).itertuples():
        print(f"  {r.test_id} {r.signal_date} [{r.impact}] {r.scenario:<34} "
              f"P&L {r.pnl / 1e6:+7.1f}m ({r.pnl_pct:+.2%})  {r.headline[:60]}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({
        "portfolio_value": float(total), "positions": int(len(portfolio)),
        "impact_threshold": IMPACT_THRESHOLD, "tests_triggered": int(len(events)),
        "company_events_without_exposure": int(events.attrs.get("no_exposure", 0)),
        "worst_test": events.iloc[0][["test_id", "signal_date", "scenario", "pnl", "pnl_pct", "headline"]].to_dict(),
        "what_if": what_if.to_dict("records"),
    }, indent=2, default=str))
    print(f"\nSaved results to {OUT_DIR} and {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module B: event-driven portfolio stress testing")
    parser.add_argument("--api", action="store_true", help="subscribe via the running Signal API")
    main(use_api=parser.parse_args().api)