"""RiskPulse dashboard: live risk signals, Module A index weights, and engine accuracy.

Reads the files produced by the pipeline and Module A, so it works offline.
Run from the repo root:   streamlit run src/dashboard.py
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "docs" / "results"

INK, PANEL = "#1F2933", "#F3F5F7"
POSITIVE, NEGATIVE, NEUTRAL, HIGH = "#127A70", "#B5384B", "#7B8794", "#C98A00"


def date_axis():
    """Monthly ticks like 'Mar 2022', so long time axes stay readable."""
    return alt.Axis(format="%b %Y", tickCount="month", labelAngle=0, title=None)

st.set_page_config(page_title="RiskPulse", page_icon="📈", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@500;700&display=swap');
h1, h2, h3 {{ font-family: 'Libre Franklin', 'Helvetica Neue', Arial, sans-serif; color: {INK}; }}
h1 {{ font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0; }}
.feed-row {{ display: flex; gap: 0.9rem; align-items: flex-start; padding: 0.55rem 0;
             border-bottom: 1px solid #E4E7EB; }}
.impact {{ flex: 0 0 2.4rem; height: 2.4rem; border-radius: 6px; display: flex;
           align-items: center; justify-content: center; font-family: 'Libre Franklin', sans-serif;
           font-weight: 700; font-size: 1.15rem; color: white; }}
.feed-meta {{ color: {NEUTRAL}; font-size: 0.85rem; }}
.feed-text {{ color: inherit; font-size: 0.98rem; line-height: 1.35; max-width: 75ch; }}
.feed-text a {{ color: inherit; text-decoration: underline; text-decoration-color: #C5CCD3; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data

@st.cache_data
def load_signals():
    path = DATA / "signals.jsonl"
    if not path.exists():
        return None
    df = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data
def load_csv(relative):
    path = DATA / relative
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_result(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def sentiment_color(value):
    if value >= 0.2:
        return POSITIVE
    if value <= -0.2:
        return NEGATIVE
    return NEUTRAL


def impact_color(value):
    return HIGH if value >= 7 else (INK if value >= 4 else NEUTRAL)


def missing(command):
    st.info(f"Nothing to show yet. Run `{command}` from the repo root, then refresh this page.")


signals = load_signals()

# ------------------------------------------------------------------ header + live feed

st.title("RiskPulse")
st.markdown("Financial risk signals from news and social media, scored for sentiment, "
            "event type and market impact.")

if signals is None:
    missing("python -m src.pipeline")
    st.stop()

live = signals[signals["source"] == "gdelt"]
feed = live if len(live) else signals
feed_title = "Highest-impact recent news" if len(live) else "Highest-impact signals"
st.subheader(feed_title)

top = feed.sort_values(["impact", "timestamp"], ascending=False).head(6)
rows = []
for r in top.itertuples():
    text = r.text if not isinstance(r.url, str) else f'<a href="{r.url}" target="_blank">{r.text}</a>'
    rows.append(
        f'<div class="feed-row"><div class="impact" style="background:{impact_color(r.impact)}">{r.impact}</div>'
        f'<div><div class="feed-text">{text}</div>'
        f'<div class="feed-meta">{r.ticker}, {r.event_type}, sentiment '
        f'<span style="color:{sentiment_color(r.sentiment)};font-weight:600">{r.sentiment:+.2f}</span>, '
        f'{r.timestamp:%d %b %Y %H:%M} UTC</div></div></div>')
st.markdown("".join(rows), unsafe_allow_html=True)
st.caption("The square is the impact score (1-10). Amber marks high impact (7 or more).")

tab_weights, tab_signals, tab_accuracy = st.tabs(["Index weights", "Signals", "How well it works"])

# ------------------------------------------------------------------ Module A

with tab_weights:
    weights = load_csv("module_a/weights_history.csv")
    curve = load_csv("module_a/equity_curve.csv")
    backtest = load_result("module_a_backtest.json")
    if weights is None or curve is None:
        missing("python -m src.modules.rebalancer")
    else:
        st.markdown("Each trading day, stocks with positive news get more weight and stocks with "
                    "negative news get less (between 2% and 15% each). The chart shows how the "
                    "15-stock index shifted over the backtest year.")
        long = weights.melt(id_vars="date", var_name="ticker", value_name="weight")
        stock_order = [c for c in weights.columns if c != "date"]
        long["tilt"] = long["weight"] - 1 / len(stock_order)
        view_mode = st.radio("Show weights as", ["Tilt versus equal weight", "Share of index"],
                             horizontal=True, label_visibility="collapsed")
        if view_mode == "Tilt versus equal weight":
            chart = alt.Chart(long).mark_rect().encode(
                x=alt.X("yearweek(date):O", title=None,
                        axis=alt.Axis(labelAngle=0, ticks=False, labelOverlap=False,
                                      labelExpr="date(datum.value) <= 7 ? timeFormat(datum.value, '%b %Y') : ''")),
                y=alt.Y("ticker:N", sort=stock_order, title=None),
                color=alt.Color("mean(tilt):Q", title="Tilt", legend=alt.Legend(format="+.0%"),
                                scale=alt.Scale(domain=[-0.025, 0, 0.025], range=[NEGATIVE, PANEL, POSITIVE],
                                                clamp=True, interpolate="rgb")),
                tooltip=[alt.Tooltip("yearweek(date):O", title="Week of", format="%d %b %Y"), "ticker:N",
                         alt.Tooltip("mean(weight):Q", title="Average weight", format=".1%"),
                         alt.Tooltip("mean(tilt):Q", title="Versus equal weight", format="+.1%")],
            ).properties(height=420)
            caption = ("Weekly average. Green: the stock was held above its equal 1/15 share because "
                       "news was positive. Red: held below it because news was negative.")
        else:
            chart = alt.Chart(long).mark_area().encode(
                x=alt.X("date:T", axis=date_axis()),
                y=alt.Y("weight:Q", stack="zero", axis=alt.Axis(format="%"), title="Share of index"),
                color=alt.Color("ticker:N", scale=alt.Scale(scheme="tableau20"), title="Stock"),
                tooltip=["date:T", "ticker:N", alt.Tooltip("weight:Q", format=".1%")],
            ).properties(height=420)
            caption = "Each band is one stock's share of the index on that day."
        st.altair_chart(chart, width="stretch")
        st.caption(caption)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Growth of 1 unit invested**")
            c = curve.rename(columns={"strategy": "Sentiment-tilted", "baseline": "Equal-weight"})
            c = c.melt(id_vars="date", var_name="portfolio", value_name="value")
            lines = alt.Chart(c).mark_line(strokeWidth=2).encode(
                x=alt.X("date:T", axis=date_axis()),
                y=alt.Y("value:Q", scale=alt.Scale(zero=False), title=None),
                color=alt.Color("portfolio:N", scale=alt.Scale(
                    domain=["Sentiment-tilted", "Equal-weight"], range=[POSITIVE, NEUTRAL]), title=None),
                tooltip=["date:T", "portfolio:N", alt.Tooltip("value:Q", format=".3f")],
            ).properties(height=280)
            st.altair_chart(lines, width="stretch")
        with right:
            if backtest:
                st.markdown("**Backtest results**")
                labels = {"total_return": "Total return", "annual_return": "Annual return",
                          "annual_volatility": "Annual volatility", "sharpe_ratio": "Sharpe ratio",
                          "max_drawdown": "Max drawdown"}
                table = pd.DataFrame({
                    "Sentiment-tilted": backtest["sentiment_tilted"],
                    "Equal-weight": backtest["equal_weight"]}).rename(index=labels)
                fmt = table.copy().astype(object)
                for idx in fmt.index:
                    is_ratio = idx == "Sharpe ratio"
                    fmt.loc[idx] = [f"{v:.2f}" if is_ratio else f"{v:+.1%}" for v in table.loc[idx]]
                st.dataframe(fmt, width="stretch")
                st.caption(f"{backtest['period'][0]} to {backtest['period'][1]}. Includes "
                           f"{backtest['settings']['cost_bps']} bps trading costs. Average daily "
                           f"turnover {backtest['avg_daily_turnover']:.1%}.")

        st.markdown("**One stock up close**")
        daily = load_csv("signals_daily.csv")
        tickers = [c for c in weights.columns if c != "date"]
        pick = st.selectbox("Stock", tickers, index=tickers.index("NFLX") if "NFLX" in tickers else 0)
        w_one = weights[["date", pick]].rename(columns={pick: "weight"})
        base = alt.Chart(w_one).encode(x=alt.X("date:T", axis=date_axis()))
        weight_line = base.mark_line(color=INK, strokeWidth=2).encode(
            y=alt.Y("weight:Q", axis=alt.Axis(format="%", title="Weight in index")))
        equal_rule = alt.Chart(pd.DataFrame({"y": [1 / len(tickers)]})).mark_rule(
            strokeDash=[4, 4], color=NEUTRAL).encode(y="y:Q")
        layers = [weight_line, equal_rule]
        if daily is not None:
            d_one = daily[(daily["ticker"] == pick) & (daily["signal_date"] <= weights["date"].max())]
            bars = alt.Chart(d_one).mark_bar(opacity=0.45).encode(
                x=alt.X("signal_date:T", axis=date_axis()),
                y=alt.Y("impact_weighted_sentiment:Q", title="Daily sentiment",
                        scale=alt.Scale(domain=[-1, 1])),
                color=alt.condition("datum.impact_weighted_sentiment >= 0",
                                    alt.value(POSITIVE), alt.value(NEGATIVE)),
                tooltip=["signal_date:T", alt.Tooltip("impact_weighted_sentiment:Q", format="+.2f"),
                         "n_texts:Q", "top_event:N"])
            chart = alt.layer(bars, alt.layer(*layers)).resolve_scale(y="independent")
        else:
            chart = alt.layer(*layers)
        st.altair_chart(chart.properties(height=280), width="stretch")
        st.caption("Line: the stock's weight (dashed = equal weight). Bars: that day's sentiment, "
                   "green positive and red negative.")

        live_w = load_csv("module_a/live_weights.csv")
        if live_w is not None:
            st.markdown(f"**Suggested weights from the latest news** (as of {live_w['as_of'].iloc[0]})")
            live_w["change"] = live_w["weight"] - live_w["equal_weight"]
            bars = alt.Chart(live_w).mark_bar().encode(
                x=alt.X("change:Q", axis=alt.Axis(format="+.0%", tickCount=6), title="Change versus equal weight"),
                y=alt.Y("ticker:N", sort="-x", title=None),
                color=alt.condition("datum.change >= 0", alt.value(POSITIVE), alt.value(NEGATIVE)),
                tooltip=["ticker:N", alt.Tooltip("weight:Q", format=".1%"),
                         alt.Tooltip("score:Q", format="+.2f")],
            ).properties(height=360)
            st.altair_chart(bars, width="stretch")
            st.caption("Stocks with no news sit slightly below equal weight only because the weights "
                       "must add up to 100% after others were raised.")

# ------------------------------------------------------------------ signals explorer

with tab_signals:
    st.markdown("Every text the engine has read, with its sentiment, event type and impact score.")
    f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
    pick_tickers = f1.multiselect("Stocks", sorted(signals["ticker"].unique()))
    pick_events = f2.multiselect("Event types", sorted(signals["event_type"].unique()))
    min_impact = f3.slider("Minimum impact", 1, 10, 5)
    pick_source = f4.radio("Source", ["All", "News", "Tweets"], horizontal=False)

    view = signals[signals["impact"] >= min_impact]
    if pick_tickers:
        view = view[view["ticker"].isin(pick_tickers)]
    if pick_events:
        view = view[view["event_type"].isin(pick_events)]
    if pick_source != "All":
        view = view[view["source"] == ("gdelt" if pick_source == "News" else "tweets")]

    st.caption(f"{len(view):,} signals match. Showing the 500 highest-impact.")
    shown = view.sort_values(["impact", "timestamp"], ascending=False).head(500)
    st.dataframe(
        shown[["timestamp", "ticker", "impact", "sentiment", "event_type", "text", "source", "url"]],
        width="stretch", hide_index=True, height=480,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM YYYY, HH:mm"),
            "ticker": "Stock",
            "impact": st.column_config.ProgressColumn("Impact", min_value=1, max_value=10, format="%d"),
            "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
            "event_type": "Event type",
            "text": st.column_config.TextColumn("Text", width="large"),
            "source": "Source",
            "url": st.column_config.LinkColumn("Link", display_text="open"),
        })

# ------------------------------------------------------------------ accuracy

with tab_accuracy:
    sent, events, impact = (load_result(n) for n in
                            ("sentiment_eval.json", "event_eval.json", "impact_eval.json"))
    st.markdown("Each part of the engine is tested on data it never saw during training.")
    rows = []
    if sent:
        rows.append(["Sentiment (FinBERT)", f"{sent['n_samples']:,} labelled finance tweets",
                     f"{sent['accuracy']:.1%} accuracy, macro-F1 {sent['macro_f1']:.2f}",
                     f"{sent['baseline_accuracy']:.1%} (always '{sent['baseline_majority_class']}')"])
    if events:
        rows.append(["Event type (TF-IDF + logistic regression)", f"{events['n_test']:,} labelled finance tweets",
                     f"{events['event_accuracy']:.1%} accuracy, macro-F1 {events['event_macro_f1']:.2f}",
                     f"{events['baseline_accuracy']:.1%} (always '{events['baseline_majority_class']}')"])
    if impact:
        rows.append(["Impact score", f"{impact['stock_days']:,} stock-days vs real prices",
                     f"rank correlation {impact['spearman_impact_vs_same_day_move']:+.3f} with same-day moves",
                     f"{impact['spearman_text_count_vs_same_day_move']:+.3f} (tweet count alone)"])
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Component", "Tested on", "Result", "Naive baseline"]),
                     width="stretch", hide_index=True)
    else:
        missing("python -m src.nlp.sentiment")

    if impact:
        st.markdown("**Do high-impact days move prices more?**")
        b = pd.DataFrame(impact["by_impact_bucket"]).rename(columns={
            "move_same_day": "Same day", "move_next_day": "Next day"})
        b = b.melt(id_vars=["impact_bucket", "stock_days"], var_name="when", value_name="move")
        chart = alt.Chart(b).mark_bar().encode(
            x=alt.X("impact_bucket:N", title="Highest impact score that day", sort=None,
                    axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("when:N", sort=["Same day", "Next day"]),
            y=alt.Y("move:Q", title="Price move vs the stock's usual day (1 = normal)"),
            color=alt.Color("when:N", scale=alt.Scale(domain=["Same day", "Next day"], range=[HIGH, NEUTRAL]),
                            title=None),
            tooltip=["impact_bucket:N", "when:N", alt.Tooltip("move:Q", format=".2f"), "stock_days:Q"],
        ).properties(height=320)
        rule = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[4, 4], color=INK).encode(y="y:Q")
        st.altair_chart(chart + rule, width="stretch")
        st.caption("Moves are measured against each stock's own typical day, so volatile stocks "
                   "don't dominate.")