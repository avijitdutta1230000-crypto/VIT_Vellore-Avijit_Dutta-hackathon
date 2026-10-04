"""Financial sentiment scoring with FinBERT (ProsusAI/finbert).

    sentiment = P(positive) - P(negative)   -> a number from -1.0 to 1.0

Run from the repo root:
    python -m src.nlp.sentiment            # accuracy test on 2,388 labeled finance tweets
    python -m src.nlp.sentiment --score    # score all tweets + GDELT news (cached, resumable)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
EVAL_CSV = ROOT / "data" / "tfns_validation.csv"
CACHE_FILE = ROOT / "data" / "processed" / "sentiment.csv"
RESULTS_FILE = ROOT / "docs" / "results" / "sentiment_eval.json"

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 96     # tweets and headlines are short, so 96 tokens is plenty
BATCH_SIZE = 32
CHUNK_SIZE = 2000   # save progress to the cache every 2,000 texts

# Twitter Financial News Sentiment labels -> FinBERT label names
TFNS_LABELS = {0: "negative", 1: "positive", 2: "neutral"}  # 0 Bearish, 1 Bullish, 2 Neutral
CLASSES = ["negative", "neutral", "positive"]

_model = None
_tokenizer = None


def load_model():
    """Load FinBERT once and reuse it (downloads ~440 MB the first time)."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def score_texts(texts, batch_size=BATCH_SIZE, progress=True):
    """Score a list of texts. Returns a DataFrame (same order) with
    'sentiment' (-1.0 to 1.0) and 'sentiment_label' (positive / negative / neutral)."""
    model, tokenizer = load_model()
    id2label = {int(i): name.lower() for i, name in model.config.id2label.items()}
    pos_idx = next(i for i, n in id2label.items() if n == "positive")
    neg_idx = next(i for i, n in id2label.items() if n == "negative")

    texts = [str(t) for t in texts]
    # Sorting by length means each batch needs less padding -> much faster on a CPU.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    scores, labels = [0.0] * len(texts), [""] * len(texts)

    for start in tqdm(range(0, len(order), batch_size), desc="FinBERT", disable=not progress):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        for row, i in enumerate(idx):
            p = probs[row]
            scores[i] = round(float(p[pos_idx] - p[neg_idx]), 4)
            labels[i] = id2label[int(p.argmax())]

    return pd.DataFrame({"sentiment": scores, "sentiment_label": labels})


def add_sentiment(df, cache_file=CACHE_FILE):
    """Add sentiment columns to a DataFrame that has 'id' and 'text' columns.

    Results are saved to a cache file as we go, so texts are never scored twice
    and an interrupted run continues where it stopped."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        cache = pd.read_csv(cache_file, dtype={"id": str})
    else:
        cache = pd.DataFrame(columns=["id", "sentiment", "sentiment_label"])

    todo = df[~df["id"].isin(cache["id"])].drop_duplicates("id")
    if len(todo):
        print(f"Scoring {len(todo):,} new texts ({df['id'].nunique() - len(todo):,} already cached)...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(todo), CHUNK_SIZE):
            chunk = todo.iloc[start:start + CHUNK_SIZE]
            scored = score_texts(chunk["text"].tolist())
            scored.insert(0, "id", chunk["id"].values)
            scored.to_csv(cache_file, mode="a", header=not cache_file.exists(), index=False)
            print(f"  saved {min(start + CHUNK_SIZE, len(todo)):,} / {len(todo):,}")
        cache = pd.read_csv(cache_file, dtype={"id": str})

    return df.merge(cache.drop_duplicates("id"), on="id", how="left")


def classification_metrics(y_true, y_pred):
    """Accuracy, macro-F1 and per-class precision / recall / F1."""
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    per_class = {}
    for c in CLASSES:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        precision = tp / max(int((y_pred == c).sum()), 1)
        recall = tp / max(int((y_true == c).sum()), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": round(precision, 3), "recall": round(recall, 3),
                        "f1": round(f1, 3), "support": int((y_true == c).sum())}
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(CLASSES)
    return round(accuracy, 4), round(macro_f1, 4), per_class


def evaluate():
    """Test FinBERT on labeled finance tweets it was NOT trained on."""
    df = pd.read_csv(EVAL_CSV)
    y_true = df["label"].map(TFNS_LABELS)
    print(f"Evaluating {MODEL_NAME} on {len(df):,} labeled tweets "
          f"(zeroshot/twitter-financial-news-sentiment, validation split)")

    pred = score_texts(df["text"].tolist())
    accuracy, macro_f1, per_class = classification_metrics(y_true, pred["sentiment_label"])

    # Naive baseline: always predict the most common label.
    majority = y_true.value_counts().idxmax()
    baseline_acc = round(float((y_true == majority).mean()), 4)

    print(f"\nAccuracy:  {accuracy:.1%}   (naive baseline, always '{majority}': {baseline_acc:.1%})")
    print(f"Macro-F1:  {macro_f1:.3f}")
    print("\nPer class:")
    print(pd.DataFrame(per_class).T.to_string())
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.crosstab(y_true.rename("actual"), pred["sentiment_label"].rename("predicted")).to_string())

    wrong = df.assign(actual=y_true, predicted=pred["sentiment_label"])
    wrong = wrong[wrong["actual"] != wrong["predicted"]]
    print("\nA few mistakes (useful for the Limitations slide):")
    for r in wrong.head(5).itertuples():
        print(f"  actual={r.actual:<8} predicted={r.predicted:<8} {r.text[:90]}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": MODEL_NAME,
        "dataset": "zeroshot/twitter-financial-news-sentiment (validation)",
        "n_samples": int(len(df)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "baseline_majority_class": majority,
        "baseline_accuracy": baseline_acc,
        "per_class": per_class,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {RESULTS_FILE}")


def score_all():
    """Score every tweet and GDELT headline, saving results to the cache."""
    from src.ingestion.gdelt import load_gdelt
    from src.ingestion.tweets import load_tweets

    frames = [f for f in (load_tweets(), load_gdelt()) if len(f)]
    df = pd.concat(frames, ignore_index=True)
    scored = add_sentiment(df)

    print("\nSentiment by source:")
    print(scored.groupby("source")["sentiment"].describe().round(3).to_string())
    print("\nAverage sentiment by ticker:")
    print(scored.groupby("ticker")["sentiment"].mean().sort_values().round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBERT sentiment scoring")
    parser.add_argument("--score", action="store_true", help="score all tweets + GDELT news")
    args = parser.parse_args()
    score_all() if args.score else evaluate()