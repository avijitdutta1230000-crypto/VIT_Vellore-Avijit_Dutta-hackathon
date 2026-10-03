# VIT_Vellore-Avijit_Dutta-hackathon
# RiskPulse: AI/NLP Financial Risk Engine - S&P Global & Crisil Campus Hackathon

<!-- Replace everything in [square brackets] and every <fill ...> before submitting. -->

**Candidate Name:** Avijit Dutta
**College Email ID:** avijit.dutta2023@vitstudent.ac.in
**College / Campus:** Vellore Institute of Technology (VIT), Vellore
**Demo Video Link:** [YouTube unlisted link]
**Slide Deck Link (if hosted externally):** Not hosted externally — see [docs/presentation.pdf](docs/presentation.pdf)

---

## 1. Project Overview / Problem Statement & Approach

Financial risk often shows up in unstructured text — a news headline about a debt downgrade, a wave of social media posts about a product recall, a report on new sanctions — before it shows up in prices or credit ratings. Risk analysts and portfolio managers cannot read thousands of articles and posts every hour, so these early signals are often missed or acted on too late. The problem is to convert this stream of text into **structured, machine-readable risk signals** that downstream systems can consume automatically.

**RiskPulse** is a unified AI/NLP Risk Engine that ingests text from multiple sources (financial news via NewsAPI, global events via GDELT, and stock-related tweets), links each item to the companies it mentions, and produces three signals per company or event: a **sentiment score** (−1.0 to 1.0) using FinBERT, an **event classification** (Geopolitical, Macroeconomic, Credit Event, M&A, Product Launch) using zero-shot classification, and an **impact score** (1–10) that combines event severity, sentiment strength and cross-source coverage. Signals are written to `data/signals.json` and served through a FastAPI endpoint.

To show the signals in action, **Module A (Tactical Index Rebalancer)** maintains a mock index of 15 S&P 100 stocks and shifts weights toward stocks with positive sentiment and away from stocks with negative sentiment, backtested against an equal-weight baseline. <!-- Delete the next sentence if you skip Module B. --> **Module B (Strategic Stress Testing)** applies predefined market shocks to a synthetic wholesale banking portfolio whenever a high-impact event (impact > 7) is detected.

## 2. Architecture & Tech Stack

![Architecture Diagram](docs/architecture.png)

### Data flow

```
NewsAPI ─┐
GDELT ───┼──> Ingestion (common schema, dedupe) ──> Entity linking (ticker aliases)
Tweets ──┘                                                   │
                                                             v
               NLP Risk Engine: FinBERT sentiment | zero-shot event class | impact score
                                                             │
                                       data/signals.json  +  FastAPI (/signals)
                                              │                          │
                                  Module A: rebalancer        Module B: stress test
                                              └──── Streamlit dashboard ───┘
```

### Tech stack

| Layer | Tool | Why |
|---|---|---|
| Ingestion | `requests`, NewsAPI, GDELT DOC API, pandas | Free sources, no paid tiers needed |
| Sentiment | FinBERT (`ProsusAI/finbert`) | Pre-trained on financial text, handles domain language better than general models |
| Event classification | Zero-shot (`facebook/bart-large-mnli`) | No labeled training data needed for custom event categories |
| Signal API | FastAPI + Uvicorn | Lightweight, auto-generated API docs |
| Market data | yfinance | Free historical prices for backtesting |
| Dashboard | Streamlit | Fast interactive charts in pure Python |

### Signal schema

| Field | Type | Description |
|---|---|---|
| `ticker` | string | Company ticker, or `MARKET` for market-wide events |
| `timestamp` | ISO 8601 | Publication time of the source text |
| `source` | string | `newsapi`, `gdelt` or `tweets` |
| `text` | string | Headline or post text |
| `sentiment` | float | −1.0 (negative) to 1.0 (positive) = P(positive) − P(negative) |
| `event_type` | string | Geopolitical, Macroeconomic, Credit Event, M&A, Product Launch, Other |
| `impact` | int | 1–10 predicted market impact |

**Impact score formula:** <!-- Update this if you change the formula. -->
`impact = clip(round(1 + 9 × event_weight × |sentiment| × coverage_factor), 1, 10)`, where `event_weight` is a fixed severity weight per event type (e.g. Credit Event > Product Launch) and `coverage_factor` grows with the number of independent sources reporting the same event.

## 3. Dataset Used

| Source | Type | Used for |
|---|---|---|
| [NewsAPI](https://newsapi.org) (free developer plan) | Public, live | Real-time financial headlines |
| [GDELT Project](https://www.gdeltproject.org) DOC API | Public, live | Global geopolitical and macro events |
| [Kaggle stock tweets dataset — add link] | Public, historical | Social media source, replayed as a stream |
| [Kaggle "Sentiment Analysis for Financial News" — add link] | Public, labeled | Evaluating sentiment accuracy |
| yfinance | Public | Daily prices for backtesting Module A |
| `data/portfolio.csv` | Synthetic (generated by `src/modules/make_portfolio.py`) | Module B banking portfolio |

**Assumptions**
- Only English-language text is processed.
- The X/Twitter API is not free, so historical tweets from Kaggle are replayed in timestamp order to simulate a live feed.
- Signals are aggregated to daily frequency; rebalancing happens once per day at the close.
- Module A ignores transaction costs and starts from equal weights, with each weight clipped between 2% and 15%.
- The Module B portfolio is fully synthetic and does not represent any real client or institution.
- `data/` contains a cached sample of every dataset used so the demo runs without API keys. Full raw datasets can be re-downloaded from the links above.

## 4. Quickstart & Installation

**Runtime:** Python 3.11 on [Windows 11 / OS you tested on]

```bash
git clone https://github.com/<your-username>/vit-guddu-hackathon.git
cd vit-guddu-hackathon
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# source venv/bin/activate        # Linux / macOS
pip install -r requirements.txt
```

> The first run downloads the FinBERT and BART models (about 2 GB).

**Option 1: Demo mode (no API keys needed, uses cached data in `data/`)**

```bash
python -m src.pipeline --demo          # builds data/signals.json from cached data
uvicorn src.api:app --port 8000        # terminal 2: signal API
streamlit run src/dashboard.py         # terminal 3: dashboard
```

**Option 2: Live mode**

```bash
cp .env.example .env                   # then add your NEWSAPI_KEY inside .env
python -m src.pipeline --live
```

**API endpoints** (interactive docs at http://localhost:8000/docs)

| Endpoint | Returns |
|---|---|
| `GET /signals?ticker=AAPL` | All signals for one ticker |
| `GET /signals/latest?limit=20` | Most recent signals across all tickers |
| `GET /health` | API status |

## 5. Key Results & Domain Impact

<!-- Fill these ONLY with numbers you actually measured. -->

| Metric | Result |
|---|---|
| FinBERT accuracy on Kaggle labeled financial news | <fill> |
| Event classification accuracy (50 hand-labeled headlines) | <fill> |
| Correlation: impact score vs. next-day absolute return | <fill> |
| Module A cumulative return vs. equal-weight baseline | <fill> vs <fill> |
| Module A max drawdown vs. equal-weight baseline | <fill> vs <fill> |

![Dashboard](docs/dashboard.png)

**Why it matters**
- **Early warning:** Credit and risk analysts get flagged on negative, high-impact events (downgrades, defaults, sanctions) as soon as they are reported, instead of after prices or ratings move.
- **Faster portfolio response:** Module A shows how sentiment signals can tilt exposure away from deteriorating names systematically rather than by manual review.
- **Event-driven stress testing:** Module B runs a stress test the moment a severe event is detected, rather than waiting for a scheduled quarterly exercise.
- **Explainability:** Every weight change and stress test traces back to the specific articles and posts that triggered it, which is essential for audit and regulatory review.

---

## Repository Structure

```
vit-guddu-hackathon/
├── README.md
├── requirements.txt
├── LICENSE
├── .env.example
├── src/
│   ├── ingestion/        # newsapi.py, gdelt.py, tweets.py
│   ├── nlp/              # sentiment.py, events.py, impact.py, entities.py
│   ├── modules/          # rebalancer.py, stress_test.py, make_portfolio.py
│   ├── pipeline.py       # runs ingestion -> NLP -> signals.json
│   ├── api.py            # FastAPI signal server
│   └── dashboard.py      # Streamlit dashboard
├── data/                 # universe.csv, cached raw samples, signals.json, portfolio.csv
└── docs/
    ├── presentation.pdf
    ├── architecture.png
    └── dashboard.png
```

## Limitations & Next Steps

- Event classification accuracy is measured on a small hand-labeled set; a larger labeled set would allow fine-tuning.
- The impact score is rule-based; a next step is training it against realized market moves.
- Free news APIs have rate limits and delays, so the system is near-real-time rather than true real-time.
- Stress test shocks are simplified fixed rules rather than calibrated historical scenarios.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
