"""
02_collect_sentiment_gdelt.py
Fetches historical headlines from GDELT for each (ticker, earnings_date)
and scores them with FinBERT. Saves data/sentiment_raw.csv.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import timedelta
from io import StringIO

os.makedirs("cache/gdelt", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ── Company name map for better GDELT search results ─────────────────────
COMPANY_NAMES = {
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "META": "Meta Facebook",
    "INTC": "Intel",
    "PLTR": "Palantir",
    "AMZN": "Amazon",
    "CRM": "Salesforce",
}

# ── Load FinBERT ──────────────────────────────────────────────────────────
print("Loading FinBERT...")
from transformers import pipeline

finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    truncation=True,
    max_length=512,
)
print("FinBERT ready.\n")


def score_headline(text):
    try:
        r = finbert([text[:512]])[0]
        if r["label"] == "positive":
            return r["score"]
        if r["label"] == "negative":
            return -r["score"]
        return 0.0
    except:
        return 0.0


def fetch_gdelt(company_name, start_date, end_date, cache_key):
    """Fetch headlines from GDELT for a company in a date window."""
    cache_file = f"cache/gdelt/{cache_key}.json"
    if os.path.exists(cache_file):
        return json.load(open(cache_file))

    query = company_name.replace(" ", "%20")
    start_s = start_date.strftime("%Y%m%d") + "000000"
    end_s = end_date.strftime("%Y%m%d") + "235959"

    url = (
        f"https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={query}%20sourcelang:eng"
        f"&mode=artlist"
        f"&maxrecords=100"
        f"&startdatetime={start_s}"
        f"&enddatetime={end_s}"
        f"&format=csv"
    )

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            json.dump([], open(cache_file, "w"))
            return []

        text = r.text.strip()
        if not text or len(text) < 50:
            json.dump([], open(cache_file, "w"))
            return []

        df = pd.read_csv(StringIO(text))
        if "title" not in df.columns and "Title" not in df.columns:
            json.dump([], open(cache_file, "w"))
            return []

        title_col = "title" if "title" in df.columns else "Title"
        articles = df[title_col].dropna().tolist()
        json.dump(articles, open(cache_file, "w"))
        return articles

    except Exception as e:
        print(f"    GDELT error: {e}")
        json.dump([], open(cache_file, "w"))
        return []


# ── Main loop ─────────────────────────────────────────────────────────────
df_prices = pd.read_csv("data/prices_raw.csv", parse_dates=["earnings_date"])
df_prices = df_prices.sort_values("earnings_date").reset_index(drop=True)

rows = []
total = len(df_prices)
skipped = 0

for idx, row in df_prices.iterrows():
    ticker = row["ticker"]
    earnings_date = pd.Timestamp(row["earnings_date"])
    start_window = earnings_date - timedelta(days=7)
    company_name = COMPANY_NAMES.get(ticker, ticker)

    cache_key = f"{ticker}_{earnings_date.strftime('%Y-%m-%d')}"
    print(f"[{idx + 1}/{total}] {ticker} {earnings_date.date()} — fetching GDELT...")

    headlines = fetch_gdelt(company_name, start_window, earnings_date, cache_key)
    time.sleep(1)  # be polite to GDELT

    if not headlines:
        print(f"  → no headlines found, skipping")
        skipped += 1
        continue

    # Score each headline
    scores = [score_headline(h) for h in headlines]
    n = len(scores)
    print(f"  → {n} headlines | mean sentiment: {np.mean(scores):.3f}")

    # Features
    sentiment_score = round(float(np.mean(scores)), 4)
    buzz_score = min(n, 100)
    mid = max(n // 2, 1)
    first_half = float(np.mean(scores[:mid]))
    second_half = float(np.mean(scores[mid:])) if n > mid else first_half
    sentiment_slope = round(second_half - first_half, 4)
    pos = sum(1 for s in scores if s > 0.1)
    neg = sum(1 for s in scores if s < -0.1)
    sentiment_polarity = round((pos - neg) / n, 4)

    rows.append(
        {
            "ticker": ticker,
            "earnings_date": earnings_date.strftime("%Y-%m-%d"),
            "sentiment_score": sentiment_score,
            "buzz_score": float(buzz_score),
            "sentiment_slope": sentiment_slope,
            "sentiment_polarity": sentiment_polarity,
            "n_headlines": n,
        }
    )

# ── Save ──────────────────────────────────────────────────────────────────
if not rows:
    print("\nERROR: No sentiment rows collected. Check GDELT connectivity.")
else:
    sent_df = pd.DataFrame(rows).sort_values("earnings_date")
    sent_df.to_csv("data/sentiment_raw.csv", index=False)

    print(f"\n=== DONE ===")
    print(f"Saved: {len(rows)} rows | Skipped: {skipped} rows")
    print(f"\nTicker coverage:")
    print(sent_df["ticker"].value_counts().to_string())
    print(f"\nSentiment variance:")
    print(
        sent_df[
            ["sentiment_score", "buzz_score", "sentiment_slope", "sentiment_polarity"]
        ]
        .describe()
        .round(3)
        .to_string()
    )
    print("\nRun 03_build_dataset.py next.")
