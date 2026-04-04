"""
02_collect_sentiment.py

Fetch financial news headlines from Finviz for each ticker, score them with
FinBERT, and save one sentiment record per (ticker, earnings_date) to
data/sentiment_raw.csv.

Features computed per (ticker, earnings_date):
  - sentiment_score: mean FinBERT score across headlines in the 7-day window
  - buzz_score: number of headlines (capped at 100)
  - sentiment_slope: shift in sentiment from first half to second half of headlines
  - sentiment_polarity: ratio of clearly positive vs negative headlines
  - n_headlines: raw count of headlines scored
"""

import json
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRICES_FILE = os.path.join("data", "prices_raw.csv")
OUTPUT_FILE = os.path.join("data", "sentiment_raw.csv")
CACHE_DIR = os.path.join("cache", "finviz")

HEADLINE_WINDOW_DAYS = 7  # look back 7 calendar days from earnings date


def load_finbert():
    """Load the FinBERT sentiment pipeline once."""
    print("Loading FinBERT model...")
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )
    print("FinBERT loaded.")
    return pipe


def fetch_and_cache_headlines(tickers: list[str]) -> None:
    """
    For each unique ticker, fetch Finviz headlines and cache to disk.
    Skips tickers that already have a cache file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Import here so the module-level import doesn't fail if not installed yet.
    from finvizfinance.quote import finvizfinance

    for ticker in tickers:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}_news.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"{ticker}: {len(cached)} headlines cached (from disk)")
            continue

        try:
            stock = finvizfinance(ticker)
            news = stock.ticker_news()

            if news is None or (isinstance(news, pd.DataFrame) and news.empty):
                records: list[dict] = []
            else:
                records = news.to_dict(orient="records")

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(records, f, default=str)

            print(f"{ticker}: {len(records)} headlines cached")

        except Exception as exc:
            print(f"ERROR fetching headlines for {ticker}: {exc}")
            # Cache an empty list so we don't retry on next run.
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        time.sleep(3)


def load_cached_headlines(ticker: str) -> pd.DataFrame:
    """Load cached headlines for *ticker* and parse dates."""
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_news.json")

    if not os.path.exists(cache_path):
        return pd.DataFrame()

    with open(cache_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Parse the Finviz date column.  finvizfinance may return dates in
    # ISO format ("2026-04-04 02:00:00") or the legacy web-scrape format
    # ("Apr-04-26 06:47PM").  Try the legacy format first, and if all
    # values come back NaT, fall back to general parsing.
    if "Date" in df.columns:
        parsed = pd.to_datetime(
            df["Date"], format="%b-%d-%y %I:%M%p", errors="coerce"
        )
        if parsed.isna().all():
            parsed = pd.to_datetime(df["Date"], errors="coerce")
        df["parsed_date"] = parsed
    else:
        df["parsed_date"] = pd.NaT

    return df


def score_one(text: str, finbert) -> float | None:
    """Score a single headline with FinBERT. Returns float or None on error."""
    try:
        result = finbert([text])[0]
        if result["label"] == "positive":
            return result["score"]
        if result["label"] == "negative":
            return -result["score"]
        return 0.0
    except Exception as exc:
        print(f"    WARNING: FinBERT error on headline, skipping — {exc}")
        return None


def compute_sentiment_features(scores: list[float]) -> dict:
    """Compute sentiment_score, buzz_score, sentiment_slope, sentiment_polarity."""
    arr = np.array(scores)

    sentiment_score = round(float(arr.mean()), 4)
    buzz_score = min(len(scores), 100)

    # Slope: second half mean minus first half mean.
    if len(scores) >= 2:
        mid = len(scores) // 2
        first_half = arr[:mid]
        second_half = arr[mid:]
        sentiment_slope = round(float(second_half.mean() - first_half.mean()), 4)
    else:
        sentiment_slope = 0.0

    # Polarity: (positive - negative) / total.
    positive_count = int((arr > 0.1).sum())
    negative_count = int((arr < -0.1).sum())
    sentiment_polarity = round((positive_count - negative_count) / len(scores), 4)

    return {
        "sentiment_score": sentiment_score,
        "buzz_score": buzz_score,
        "sentiment_slope": sentiment_slope,
        "sentiment_polarity": sentiment_polarity,
    }


def main() -> None:
    # --- Load inputs ------------------------------------------------------
    if not os.path.exists(PRICES_FILE):
        print(f"ERROR: {PRICES_FILE} not found. Run 01_collect_prices.py first.")
        return

    prices_df = pd.read_csv(PRICES_FILE)
    prices_df["earnings_date"] = pd.to_datetime(prices_df["earnings_date"]).dt.date
    unique_tickers = prices_df["ticker"].unique().tolist()

    print(f"Loaded {len(prices_df)} rows from {PRICES_FILE}")
    print(f"Unique tickers: {unique_tickers}\n")

    # --- Step 1: Load FinBERT ---------------------------------------------
    finbert = load_finbert()

    # --- Step 2: Fetch & cache headlines per ticker -----------------------
    fetch_and_cache_headlines(unique_tickers)
    print()

    # --- Step 3: Score headlines per (ticker, earnings_date) ---------------
    rows: list[dict] = []
    total = len(prices_df)

    for i, row in prices_df.iterrows():
        ticker = row["ticker"]
        edate = row["earnings_date"]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing row {i + 1}/{total}: {ticker} {edate}")

        headlines_df = load_cached_headlines(ticker)

        if headlines_df.empty or "Title" not in headlines_df.columns:
            print(f"  SKIP {ticker} {edate} — no headlines in window")
            continue

        # Filter to 7-day window before earnings (inclusive).
        edate_ts = pd.Timestamp(edate)
        window_start = edate_ts - timedelta(days=HEADLINE_WINDOW_DAYS)
        mask = (
            headlines_df["parsed_date"].notna()
            & (headlines_df["parsed_date"].dt.normalize() >= window_start)
            & (headlines_df["parsed_date"].dt.normalize() <= edate_ts)
        )
        window_df = headlines_df.loc[mask].copy()

        if window_df.empty:
            print(f"  SKIP {ticker} {edate} — no headlines in window")
            continue

        # Sort chronologically for slope computation.
        window_df = window_df.sort_values("parsed_date")

        # Score each headline.
        scores: list[float] = []
        for title in window_df["Title"]:
            if not isinstance(title, str) or not title.strip():
                continue
            s = score_one(title, finbert)
            if s is not None:
                scores.append(s)

        if not scores:
            print(f"  SKIP {ticker} {edate} — no valid scores after FinBERT")
            continue

        features = compute_sentiment_features(scores)
        rows.append({
            "ticker": ticker,
            "earnings_date": edate,
            **features,
            "n_headlines": len(scores),
        })

    # --- Step 4: Save -----------------------------------------------------
    if not rows:
        print("\nNo valid sentiment rows collected. Nothing to save.")
        return

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("earnings_date").reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {OUTPUT_FILE}")
    print(f"Shape: {out_df.shape}")

    print(f"\nPer-column std (confirms feature variance):")
    numeric_cols = ["sentiment_score", "buzz_score", "sentiment_slope", "sentiment_polarity"]
    for col in numeric_cols:
        print(f"  {col}: {out_df[col].std():.4f}")

    total_pairs = len(prices_df)
    sentiment_pairs = len(out_df)
    print(f"\nSentiment coverage: {sentiment_pairs}/{total_pairs} "
          f"({100 * sentiment_pairs / total_pairs:.1f}%) of (ticker, earnings_date) pairs")


if __name__ == "__main__":
    main()
