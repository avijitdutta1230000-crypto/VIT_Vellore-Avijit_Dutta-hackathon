"""Event classification: what kind of event is each text about?

Approach: train a fast TF-IDF + logistic regression classifier on the labelled
zeroshot/twitter-financial-news-topic dataset (~17K finance tweets, 20 topics),
then group the 20 topics into our event classes (Geopolitical, Macroeconomic,
Credit Event, M&A, Earnings, ...).

Why not a zero-shot transformer? On a laptop CPU it would need many hours for
~57K tweets. This classifier trains in about a minute, labels everything in
seconds, and we can measure its accuracy on held-out labelled data.

Run from the repo root:
    python -m src.nlp.events            # train + evaluate on the held-out validation split
    python -m src.nlp.events --score    # label all tweets + GDELT news (cached)
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.ingestion.common import ROOT, clean_text

TRAIN_CSV = ROOT / "data" / "tfn_topic_train.csv"
VALID_CSV = ROOT / "data" / "tfn_topic_validation.csv"
MODEL_FILE = ROOT / "models" / "event_classifier.joblib"
CACHE_FILE = ROOT / "data" / "processed" / "events.csv"
RESULTS_FILE = ROOT / "docs" / "results" / "event_eval.json"

# Label ids of zeroshot/twitter-financial-news-topic (from its dataset card).
TOPICS = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News",
    "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil",
    "Financials", "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation",
    "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change",
    "Stock Commentary", "Stock Movement",
]

# How the 20 fine-grained topics map onto our event classes.
TOPIC_TO_EVENT = {
    "Politics": "Geopolitical",
    "Macro": "Macroeconomic",
    "Fed | Central Banks": "Macroeconomic",
    "Currencies": "Macroeconomic",
    "Energy | Oil": "Macroeconomic",
    "Gold | Metals | Materials": "Macroeconomic",
    "Treasuries | Corporate Debt": "Credit Event",
    "M&A | Investments": "Merger/Acquisition",
    "Company | Product News": "Product/Company News",
    "Earnings": "Earnings",
    "Dividend": "Earnings",
    "Analyst Update": "Analyst Rating",
    "Legal | Regulation": "Legal/Regulatory",
    "IPO": "Other",
    "Financials": "Other",
    "Personnel Change": "Other",
    "General News | Opinion": "Other",
    "Markets": "Other",
    "Stock Commentary": "Other",
    "Stock Movement": "Other",
}
EVENTS = sorted(set(TOPIC_TO_EVENT.values()))

CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,6}\b")


def normalize(text):
    """Clean text and replace cashtags with one token, so the classifier learns
    the kind of event, not which ticker happened to appear in the training data."""
    return CASHTAG_RE.sub(" cashtag ", clean_text(text))


def load_topic_data():
    """Load the labelled topic dataset (downloads it from Hugging Face the first time)."""
    if not (TRAIN_CSV.exists() and VALID_CSV.exists()):
        from datasets import load_dataset
        print("Downloading zeroshot/twitter-financial-news-topic ...")
        ds = load_dataset("zeroshot/twitter-financial-news-topic")
        ds["train"].to_pandas().to_csv(TRAIN_CSV, index=False)
        ds["validation"].to_pandas().to_csv(VALID_CSV, index=False)
    return pd.read_csv(TRAIN_CSV), pd.read_csv(VALID_CSV)


def build_model():
    """Word + character n-gram TF-IDF features -> logistic regression."""
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=3000, C=5.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def train(train_df):
    model = build_model()
    model.fit(train_df["text"].map(normalize), train_df["label"])
    return model


def get_model():
    """Load the saved classifier, or train and save it if there isn't one yet."""
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    train_df, _ = load_topic_data()
    print(f"Training event classifier on {len(train_df):,} labelled tweets...")
    model = train(train_df)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    return model


def predict(model, texts):
    """Returns a DataFrame (same order) with topic, event_type and event_confidence.

    The event probability is the sum of the probabilities of its topics, e.g.
    P(Macroeconomic) = P(Macro) + P(Fed) + P(Currencies) + P(Energy) + P(Gold)."""
    texts = [normalize(t) for t in texts]
    if not texts:
        return pd.DataFrame(columns=["topic", "event_type", "event_confidence"])
    probs = model.predict_proba(texts)                      # shape: (n_texts, n_topics)
    topic_ids = model.classes_
    event_probs = np.zeros((len(texts), len(EVENTS)))
    for col, topic_id in enumerate(topic_ids):
        event_probs[:, EVENTS.index(TOPIC_TO_EVENT[TOPICS[topic_id]])] += probs[:, col]
    best_event = event_probs.argmax(axis=1)
    return pd.DataFrame({
        "topic": [TOPICS[topic_ids[i]] for i in probs.argmax(axis=1)],
        "event_type": [EVENTS[i] for i in best_event],
        "event_confidence": event_probs.max(axis=1).round(4),
    })


def metrics(y_true, y_pred, classes):
    """Accuracy, macro-F1 and per-class precision / recall / F1."""
    y_true, y_pred = pd.Series(list(y_true)), pd.Series(list(y_pred))
    per_class = {}
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        precision = tp / max(int((y_pred == c).sum()), 1)
        recall = tp / max(int((y_true == c).sum()), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": round(precision, 3), "recall": round(recall, 3),
                        "f1": round(f1, 3), "support": int((y_true == c).sum())}
    accuracy = round(float((y_true == y_pred).mean()), 4)
    macro_f1 = round(sum(m["f1"] for m in per_class.values()) / len(classes), 4)
    return accuracy, macro_f1, per_class


def evaluate():
    """Train on the train split, test on the held-out validation split."""
    train_df, valid_df = load_topic_data()

    print("Label check (one example per topic id - they should match the topic name):")
    for topic_id, name in enumerate(TOPICS):
        sample = train_df[train_df["label"] == topic_id]["text"].head(1)
        text = clean_text(sample.iloc[0])[:85] if len(sample) else "(no examples)"
        print(f"  {topic_id:>2} {name:<27} | {text}")

    print(f"\nTraining on {len(train_df):,} tweets, testing on {len(valid_df):,} held-out tweets...")
    model = train(train_df)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    pred = predict(model, valid_df["text"].tolist())
    true_topics = valid_df["label"].map(lambda i: TOPICS[i])
    true_events = true_topics.map(TOPIC_TO_EVENT)

    topic_acc, topic_f1, _ = metrics(true_topics, pred["topic"], TOPICS)
    event_acc, event_f1, per_event = metrics(true_events, pred["event_type"], EVENTS)
    majority = true_events.value_counts().idxmax()
    baseline_acc = round(float((true_events == majority).mean()), 4)

    print(f"\n20-topic accuracy:   {topic_acc:.1%}   (macro-F1 {topic_f1:.3f})")
    print(f"Event-class accuracy: {event_acc:.1%}   (macro-F1 {event_f1:.3f})")
    print(f"Naive baseline (always '{majority}'): {baseline_acc:.1%}")
    print("\nPer event class:")
    print(pd.DataFrame(per_event).T.to_string())

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({
        "model": "TF-IDF (word 1-2 + char 3-5 grams) + logistic regression",
        "dataset": "zeroshot/twitter-financial-news-topic (train -> validation)",
        "n_train": int(len(train_df)), "n_test": int(len(valid_df)),
        "topic_accuracy": topic_acc, "topic_macro_f1": topic_f1,
        "event_accuracy": event_acc, "event_macro_f1": event_f1,
        "baseline_majority_class": majority, "baseline_accuracy": baseline_acc,
        "per_event": per_event, "topic_to_event": TOPIC_TO_EVENT,
    }, indent=2))
    print(f"\nSaved results to {RESULTS_FILE}")


def add_events(df, cache_file=CACHE_FILE):
    """Add topic / event_type / event_confidence to a DataFrame with 'id' and 'text',
    using a cache so texts are only classified once."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        cache = pd.read_csv(cache_file, dtype={"id": str})
    else:
        cache = pd.DataFrame(columns=["id", "topic", "event_type", "event_confidence"])

    todo = df[~df["id"].isin(cache["id"])].drop_duplicates("id")
    if len(todo):
        print(f"Classifying {len(todo):,} new texts ({df['id'].nunique() - len(todo):,} already cached)...")
        labels = predict(get_model(), todo["text"].tolist())
        labels.insert(0, "id", todo["id"].values)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(cache_file, mode="a", header=not cache_file.exists(), index=False)
        cache = pd.read_csv(cache_file, dtype={"id": str})

    return df.merge(cache.drop_duplicates("id"), on="id", how="left")


def score_all():
    """Classify every tweet and GDELT headline, saving results to the cache."""
    from src.ingestion.gdelt import load_gdelt
    from src.ingestion.tweets import load_tweets

    frames = [f for f in (load_tweets(), load_gdelt()) if len(f)]
    labelled = add_events(pd.concat(frames, ignore_index=True))

    print("\nEvent types by source (% of texts):")
    print((pd.crosstab(labelled["event_type"], labelled["source"], normalize="columns") * 100).round(1).to_string())
    print("\nExamples (most confident per event type):")
    for event in EVENTS:
        rows = labelled[labelled["event_type"] == event].nlargest(2, "event_confidence")
        for r in rows.itertuples():
            print(f"  [{event}] ({r.event_confidence:.2f}) {r.text[:90]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event classification")
    parser.add_argument("--score", action="store_true", help="label all tweets + GDELT news")
    args = parser.parse_args()
    score_all() if args.score else evaluate()