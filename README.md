# RiskPulse: AI/NLP Financial Risk Engine - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Avijit Dutta
**College Email ID:** [your_id@vitstudent.ac.in]
**College / Campus:** Vellore Institute of Technology (VIT), Vellore
**Demo Video Link:** [YouTube unlisted link]
**Slide Deck Link (if hosted externally):** Not hosted externally, see [docs/presentation.pdf](docs/presentation.pdf)

---

## 1. Project Overview / Problem Statement & Approach

Financial risk usually shows up in text before it shows up in prices or ratings: a downgrade, a lawsuit, a Fed comment, a sanctions headline. Risk teams cannot read thousands of news articles and posts every hour, so these early signals are missed or acted on late. The task was to build an engine that turns this unstructured stream into **structured, machine-readable risk signals**, and to show two practical uses of them.

**RiskPulse** ingests text from two sources: about 80,000 stock tweets (replayed in time order as a social media feed) and live global news from the GDELT Project. Every text is linked to the company it is really about, then scored three ways: **sentiment** from -1 to +1 with FinBERT, an **event type** (Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Earnings, Analyst Rating, Legal/Regulatory, Product/Company News, Other) with a classifier trained on labelled finance tweets, and an **impact score** from 1 to 10 that combines event severity, classifier confidence, sentiment strength and abnormal attention. Each part is tested on data it never saw, and the impact score is checked against real price moves.

The signals are written to files and served by a FastAPI **Signal API**, which both downstream modules use. **Module A** tilts a 15-stock S&P 100 index towards stocks with positive news and backtests it against equal weight. **Module B** stress-tests a synthetic $862m wholesale banking book whenever a high-impact event (score 7 or more) is detected, using rate, currency, commodity, geopolitical, credit and company-specific shock scenarios. A Streamlit dashboard shows the live risk feed, the index weights over time, the stress results and the accuracy of every component.

## 2. Architecture & Tech Stack

![Architecture Diagram](docs/architecture.png)

**Data flow**

1. **Ingestion** (`src/ingestion/`): tweets and GDELT news are cleaned (links removed, spam duplicates dropped) and put into one common schema: `id, source, timestamp, ticker, text, url`. Market-wide news is tagged `MARKET`.
2. **Entity linking** (`src/ingestion/common.py`): a text counts for a company only if it names it (name, alias or cashtag, including old tickers such as `$FB` for Meta). This also fixed a bug in the source data, where the same 4,080 tweets were filed under three different companies.
3. **Sentiment** (`src/nlp/sentiment.py`): FinBERT (`ProsusAI/finbert`), score = P(positive) - P(negative). Results are cached so each text is scored only once.
4. **Event type** (`src/nlp/events.py`): TF-IDF (word and character n-grams) + logistic regression, trained on 17K labelled finance tweets with 20 topics, grouped into 9 event classes.
5. **Impact score** (`src/nlp/impact.py`): `1 + 9 x (event weight x confidence factor x sentiment strength x attention)`, where attention compares today's number of texts about a stock with its own past 30 days (past data only, so no look-ahead).
6. **Signal output** (`src/pipeline.py`): `data/signals.jsonl` (one signal per text) and `data/signals_daily.csv` (one row per stock per day), served by `src/api.py`.
7. **Module A** (`src/modules/rebalancer.py`) subscribes to daily sentiment; **Module B** (`src/modules/stress_test.py`) subscribes to high-impact events. Both can read from the files or from the running API (`--api`).
8. **Dashboard** (`src/dashboard.py`).

**Tech stack**

| Layer | Tools | Why |
|---|---|---|
| Ingestion | pandas, requests, GDELT DOC 2.0 API | Free, no API key, global news every 15 minutes |
| Sentiment | PyTorch, Hugging Face Transformers, FinBERT | Pre-trained on financial language |
| Event type | scikit-learn (TF-IDF + logistic regression) | Trains in about a minute and labels 57K texts in seconds on a CPU; a zero-shot transformer would take hours |
| Signal API | FastAPI, Uvicorn | Simple REST endpoints with automatic interactive docs at `/docs` |
| Modules | pandas, NumPy | Backtest and portfolio revaluation |
| Dashboard | Streamlit, Altair | Interactive charts in pure Python |

**Repository structure**

```
VIT_Vellore-Avijit_Dutta-hackathon/
├── README.md
├── requirements.txt
├── LICENSE
├── .streamlit/config.toml     # dashboard theme
├── src/
│   ├── ingestion/             # common.py (schema, entity linking), tweets.py, gdelt.py
│   ├── nlp/                   # sentiment.py, events.py, impact.py
│   ├── modules/               # rebalancer.py (Module A), stress_test.py (Module B)
│   ├── pipeline.py            # runs the whole engine
│   ├── api.py                 # Signal API
│   └── dashboard.py           # Streamlit dashboard
├── data/                      # all data used (see section 3)
└── docs/
    ├── presentation.pdf
    ├── architecture.png
    └── results/               # evaluation results as JSON
```

## 3. Dataset Used

All data is public or synthetic. No real client or confidential data is used.

| File | Source | Used for |
|---|---|---|
| `data/raw/stock_tweets.csv` | Kaggle, "Stock Tweets for Sentiment Analysis and Prediction" (equinxx), Sep 2021 - Sep 2022 | Social media source, replayed in time order |
| `data/raw/stock_yfinance_data.csv` | Same Kaggle dataset, daily prices | Module A backtest, impact validation |
| `data/raw/gdelt.jsonl` | GDELT DOC 2.0 API (cached headlines) | Live news source |
| `data/tfns_validation.csv` | Hugging Face `zeroshot/twitter-financial-news-sentiment` (MIT) | Testing FinBERT |
| `data/tfn_topic_*.csv` | Hugging Face `zeroshot/twitter-financial-news-topic` (MIT) | Training and testing the event classifier |
| `data/universe.csv` | Built for this project | The 15 S&P 100 stocks and their aliases |
| `data/module_b/portfolio.csv` | Synthetic, generated by `stress_test.py` | Module B banking book (fictional) |
| `data/processed/`, `data/signals_daily.csv`, `data/module_a/`, `data/module_b/` | Produced by the code | Cached engine outputs, so the demo runs offline |

**Assumptions**

- Only English text is processed.
- The X/Twitter API is not free, so historical tweets replace a live social feed. The Kaggle tweets and prices cover Sep 2021 - Sep 2022; GDELT provides the live part.
- NewsAPI was dropped because signing up for a key failed; tweets and GDELT already meet the two-source requirement.
- Texts posted after the 4 pm New York close count towards the next trading day.
- The stock universe is 15 S&P 100 stocks that have real tweet coverage. Procter & Gamble was replaced by Salesforce because only 27 genuine P&G tweets remained after the duplicate fix.
- Module A trades at the close, holds weights for the next day, keeps each stock between 2% and 15%, and pays 10 bps on every change in weight.
- Module B uses first-order sensitivities (beta, duration, spread duration, expected loss) and shock sizes calibrated for an impact-8 event, scaled by impact.

## 4. Quickstart & Installation

**Runtime:** Python [3.x, run `python --version`] on Windows 11 (Git Bash). Should also work on Linux and macOS.

```bash
git clone https://github.com/<your-username>/VIT_Vellore-Avijit_Dutta-hackathon.git
cd VIT_Vellore-Avijit_Dutta-hackathon
python -m venv venv
source venv/Scripts/activate        # Windows (Git Bash)
# source venv/bin/activate          # Linux / macOS
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Run the demo (uses the cached data in `data/`, no API keys, works offline):**

```bash
python -m src.pipeline                 # builds the signals (about 1 minute)
python -m src.modules.rebalancer       # Module A backtest
python -m src.modules.stress_test      # Module B stress tests
streamlit run src/dashboard.py         # dashboard at http://localhost:8501
```

**Signal API** (in a second terminal):

```bash
uvicorn src.api:app --port 8000        # interactive docs at http://localhost:8000/docs
python -m src.modules.rebalancer --api # Module A subscribing through the API
python -m src.modules.stress_test --api
```

**Refresh with live news** (internet needed; FinBERT downloads about 440 MB the first time):

```bash
python -m src.ingestion.gdelt          # about 5 minutes, GDELT allows one request every few seconds
python -m src.pipeline
```

**Reproduce the accuracy tests:**

```bash
python -m src.nlp.sentiment            # FinBERT on held-out labelled tweets
python -m src.nlp.events               # event classifier on the held-out validation split
```

## 5. Key Results & Domain Impact

**Every part of the engine was tested on data it never saw**

| Component | Tested on | Result | Naive baseline |
|---|---|---|---|
| Sentiment (FinBERT) | 2,388 labelled finance tweets | **72.5% accuracy, macro-F1 0.67** | 65.6% accuracy, macro-F1 about 0.26 (always "neutral") |
| Event type | 4,117 labelled finance tweets | **89.0% accuracy, macro-F1 0.87** | 35.8% (always "Other") |
| Impact score | about 3,000 stock-days against real prices | **Score 7-10 days moved 2.9x normal the same day** | Score 1-3 days: 0.96x |

FinBERT was deliberately *not* tested on Financial PhraseBank, because it was trained on it and the score would be inflated.

**What the engine produced:** 56,931 signals from 56,819 cleaned tweets (80,793 raw) and the GDELT headlines. The impact score ranks price moves better than simply counting tweets (rank correlation 0.085 vs 0.059).

**Module A, sentiment-tilted index (Oct 2021 - Sep 2022):** -32.9% a year against -32.3% for equal weight, in a severe tech bear market. Before trading costs the two were identical (+0.03% a year), and all nine robustness settings tell the same story. The impact check explains why: high-impact days move 2.9x normal on the same day but only 1.5x the next day, because most tweets react to moves that already happened. **The engine is a strong real-time risk monitor, not a next-day trading signal.**

**Module B, stress testing:** 40 stress tests were triggered by high-impact events (15 market-wide, 25 company-specific; 14 more company events were skipped because the book holds nothing in them). The triggers are real 2022 events, such as Chair Powell signalling a 50 bp hike (21 Apr 2022, a rate-shock scenario) and Disney's Florida district debt story (an issuer credit scenario). The what-if analysis shows a **+2% rate shock costs this book about 10%**, four times more than a geopolitical shock, because two thirds of it is bonds and loans.

**Why it matters**

- **Early warning for risk teams:** analysts get a ranked feed of the events that move prices, with the company, event type and severity already extracted, instead of reading thousands of posts.
- **Event-driven stress testing:** a bank can see what a breaking event would do to its book the moment it is reported, instead of waiting for a scheduled stress exercise.
- **Explainability:** every weight change and every stress test traces back to the exact headline or tweet that triggered it, which matters for audit and regulators.
- **Honest evaluation:** each model is compared with a naive baseline, and results that did not work (Module A's next-day edge) are reported, not tuned away.

## 6. Limitations & Next Steps

- FinBERT reads analyst jargon ("cuts to Equal Weight") as neutral; fine-tuning it on labelled finance tweets would likely help.
- The event classifier is word-based, so slang can fool it ("Gold secured! $TSLA" was labelled Macroeconomic).
- The impact score uses hand-set event weights; the next step is learning them from realised price moves.
- Only 49 stock-days scored 7 or more, so the price-move result is strong evidence but not proof.
- Module B shocks are fixed per scenario and scaled by impact; a real bank would size each shock to the specific event and use full repricing.
- The free GDELT API rate-limits heavy use, so the live feed is near-real-time rather than streaming.

## 7. AI Assistance

[Describe honestly how you used AI tools, for example: "An AI assistant (Claude) was used to help draft and debug code and documentation. All design decisions, runs, results and their interpretation were reviewed by me."]

## License

MIT, see [LICENSE](LICENSE).