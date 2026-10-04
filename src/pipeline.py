"""End-to-end Risk Engine pipeline: text -> structured risk signals.

    tweets + GDELT news -> FinBERT sentiment -> event classification -> impact score
    -> data/signals.jsonl        (one signal per text, for the API and dashboard)
    -> data/signals_daily.csv    (one row per stock per day, for Module A)
    -> docs/results/impact_eval.json  (impact score vs real price moves)

Sentiment and event labels are cached, so after the first run this works
offline in about a minute.

Run from the repo root:  python -m src.pipeline
"""
import json

import pandas as pd

from src.ingestion.common import ROOT
from src.ingestion.gdelt import load_gdelt
from src.ingestion.tweets import load_tweets
from src.nlp.events import add_events
from src.nlp.impact import add_impact, validate
from src.nlp.sentiment import add_sentiment

SIGNALS_FILE = ROOT / "data" / "signals.jsonl"
DAILY_FILE = ROOT / "data" / "signals_daily.csv"
IMPACT_RESULTS = ROOT / "docs" / "results" / "impact_eval.json"

SIGNAL_COLUMNS = ["id", "timestamp", "signal_date", "source", "ticker", "text", "url",
                  "sentiment", "sentiment_label", "topic", "event_type", "event_confidence",
                  "attention_ratio", "impact"]


def build_signals():
    """Run every stage of the engine and return one row per text."""
    frames = [f for f in (load_tweets(), load_gdelt()) if len(f)]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} texts: " + ", ".join(f"{s} {n:,}" for s, n in df["source"].value_counts().items()))
    df = add_sentiment(df)
    df = add_events(df)
    df = add_impact(df)
    return df[SIGNAL_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def daily_summary(signals):
    """One row per stock per day: the input Module A rebalances on.

    impact_weighted_sentiment gives important news more say than chatter."""
    s = signals.assign(weighted=signals["sentiment"] * signals["impact"])
    top = s.loc[s.groupby(["ticker", "signal_date"])["impact"].idxmax(), ["ticker", "signal_date", "event_type"]]
    daily = s.groupby(["ticker", "signal_date"]).agg(
        n_texts=("id", "size"),
        mean_sentiment=("sentiment", "mean"),
        weighted_sum=("weighted", "sum"),
        impact_sum=("impact", "sum"),
        max_impact=("impact", "max"),
        n_high_impact=("impact", lambda x: int((x >= 7).sum())),
    ).reset_index()
    daily["impact_weighted_sentiment"] = daily["weighted_sum"] / daily["impact_sum"]
    daily = daily.merge(top.rename(columns={"event_type": "top_event"}), on=["ticker", "signal_date"])
    cols = ["ticker", "signal_date", "n_texts", "mean_sentiment", "impact_weighted_sentiment",
            "max_impact", "n_high_impact", "top_event"]
    return daily[cols].round(4).sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def main():
    signals = build_signals()
    signals.assign(signal_date=signals["signal_date"].astype(str)).to_json(
        SIGNALS_FILE, orient="records", lines=True, date_format="iso")
    daily = daily_summary(signals)
    daily.to_csv(DAILY_FILE, index=False)
    print(f"\nWrote {len(signals):,} signals to {SIGNALS_FILE.name} and "
          f"{len(daily):,} stock-days to {DAILY_FILE.name}")

    print("\nImpact score distribution:")
    print(signals["impact"].value_counts().sort_index().to_string())
    print("\nAverage impact by event type:")
    print(signals.groupby("event_type")["impact"].agg(["mean", "max", "size"]).round(2)
          .sort_values("mean", ascending=False).to_string())

    print("\nHighest-impact signals:")
    for r in signals.nlargest(5, ["impact", "event_confidence"]).itertuples():
        print(f"  [{r.impact}] {r.ticker:<6} {r.event_type:<20} sent={r.sentiment:+.2f}  {r.text[:70]}")

    results = validate(signals)
    if results:
        IMPACT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        IMPACT_RESULTS.write_text(json.dumps(results, indent=2))
        print("\nDoes impact line up with real price moves? (moves relative to each stock's usual day)")
        print(pd.DataFrame(results["by_impact_bucket"]).to_string(index=False))
        print(f"Rank correlation, impact vs same-day move:  {results['spearman_impact_vs_same_day_move']}")
        print(f"Rank correlation, impact vs next-day move:  {results['spearman_impact_vs_next_day_move']}")
        print(f"Rank correlation, text count vs same-day move (baseline): "
              f"{results['spearman_text_count_vs_same_day_move']}")
        print(f"Saved to {IMPACT_RESULTS}")


if __name__ == "__main__":
    main()