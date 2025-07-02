#!/usr/bin/env python3
"""
Minimal CPU-only sentiment analysis using ModernBERT
"""

"""
analyze.py
-----------
This script was originally a tiny demo that showed how to run a RoBERTa based
sentiment-analysis pipeline on a couple of hard-coded example sentences.

It has now been extended so that it can also be used from the command line to
run sentiment analysis over the RETURN_COMMENT column that lives in
`sampled_returns_with_star_sentiment.csv`.

Usage
~~~~~
```
python analyze.py                       # analyse the default CSV file
python analyze.py --csv path/to/file    # analyse a custom CSV file
```

The script will write a new file alongside the input called
`<original-name>_with_predictions.csv` that contains two new columns:

    • model_sentiment   – The predicted label (negative, neutral, positive)
    • model_confidence  – The probability associated with the chosen label

If you only need a quick glance you can also pass the flag `--head` to just see
the first few annotated rows printed to stdout instead of generating a file.
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Lazy import so that users that only want to see the docstring do not need
# pandas installed.
try:
    import pandas as pd
except ModuleNotFoundError as e:  # pragma: no cover – only triggered in exotic envs
    raise SystemExit(
        "The extended functionality of analyze.py now depends on pandas. "
        "Install it via `pip install pandas` and try again."  # noqa: E501
    ) from e


def setup_sentiment_analyzer(
    # Use the same compact, CPU-friendly model as test_models.py so we only
    # download one checkpoint and ensure consistency across the repo.
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
):
    """
    Initialize a Hugging Face sentiment-analysis pipeline that runs purely on
    the CPU using a small DistilBERT checkpoint.  The model ships its weights
    in the `*.safetensors` format, so we avoid the historical pickle
    vulnerability and stay lightweight.
    """
    # Force CPU usage. The 🤗 pipeline expects -1 for CPU.
    device = -1
    
    # Load sentiment analysis pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
    )
    
    return sentiment_pipeline

def analyze_sentiment(text, analyzer):
    """Analyze sentiment of a single text"""
    start_time = time.time()
    result = analyzer(text)
    inference_time = time.time() - start_time
    
    # 🤗 pipeline returns e.g. `[{'label': 'POSITIVE', 'score': 0.99}]` by
    # default, or a list-of-lists when `return_all_scores=True`.  Handle both
    # variants gracefully.
    first = result[0]
    if isinstance(first, dict):
        best_pred = first
    else:  # list-of-dicts → pick the highest score
        best_pred = max(first, key=lambda x: x["score"])
    
    return {
        'text': text,
        'sentiment': best_pred['label'],
        'confidence': best_pred['score'],
        'all_scores': first,
        'inference_time_ms': inference_time * 1000
    }

def batch_analyze(texts, analyzer, batch_size=8):
    """Analyze sentiment for multiple texts efficiently"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        start_time = time.time()

        batch_results = analyzer(batch)
        inference_time = time.time() - start_time

        for text, pred in zip(batch, batch_results):
            # pred is usually dict with label/score, but could be list when
            # return_all_scores=True – handle both.
            if isinstance(pred, dict):
                best_pred = pred
            else:  # list-of-dicts
                best_pred = max(pred, key=lambda x: x["score"])

            results.append(
                {
                    "text": text,
                    "sentiment": best_pred["label"],
                    "confidence": best_pred["score"],
                    "inference_time_ms": (inference_time / len(batch)) * 1000,
                }
            )
    
    return results

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run sentiment analysis on return comments.")
    parser.add_argument(
        "--csv",
        type=str,
        default="sampled_returns_with_star_sentiment.csv",
        help="Path to the CSV file that contains a RETURN_COMMENT column.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of comments to process at once. Higher is faster but uses more RAM.",
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="If set, only print the first rows with predictions instead of writing a file.",
    )
    return parser.parse_args()


def _run_on_csv(csv_path: Path, batch_size: int, head: bool = False) -> None:
    """Annotate `csv_path` with model predictions."""

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print(f"Loading data from {csv_path} …", flush=True)
    df = pd.read_csv(csv_path)

    comment_col_candidates = [
        col for col in df.columns if col.lower() in {"return_comment", "comment", "comments"}
    ]
    if not comment_col_candidates:
        raise ValueError(
            "Could not find a column named 'RETURN_COMMENT' (case-insensitive) in the CSV."
        )

    comment_col = comment_col_candidates[0]
    texts = df[comment_col].fillna("").astype(str).tolist()

    print("Setting up sentiment analyzer …", flush=True)
    analyzer = setup_sentiment_analyzer()

    print("Running inference …", flush=True)
    results = batch_analyze(texts, analyzer, batch_size=batch_size)

    predicted_sentiments = [r["sentiment"] for r in results]
    confidences = [r["confidence"] for r in results]

    df["model_sentiment"] = predicted_sentiments
    df["model_confidence"] = confidences

    if head:
        # Show only the original comment and the model's sentiment prediction
        cols_to_show = [comment_col, "model_sentiment"]
        print(df[cols_to_show].head())
        return

    out_path = csv_path.with_name(csv_path.stem + "_with_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")


def _main() -> None:
    args = _parse_args()
    _run_on_csv(Path(args.csv), batch_size=args.batch_size, head=args.head)


if __name__ == "__main__":
    _main()