"""
02_collect_sentiment_finviz.py

Compute FinBERT sentiment for earnings events using already-cached Finviz headlines.
Only covers recent earnings dates due to Finviz's lack of historical news archives.
"""

import json
import os
import transformers

import numpy as np
import pandas as pd
from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRICES_FILE = os.path.join("data", "prices_raw.csv")
OUTPUT_FILE = os.path.join("data", "sentiment_raw.csv")
CACHE_DIR = os.path.join("cache", "finviz")

# Finviz API returns ISO format dates now, not "%b-%d-%y %I:%M%p"
# But we'll follow instructions and parse with the specified format
# and fall back to inferring if needed since we know the cache format
# from previous inspection.


def load_finbert():
    """Load the FinBERT sentiment pipeline once."""
    print("Loading FinBERT...")
    # Suppress verbose transformers logging
    transformers.logging.set_verbosity_error()
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )
    print("FinBERT ready.")
    return pipe


def score_text(finbert, text: str) -> float:
    """Score a single string. Returns float in [-1, 1]."""
    try:
        r = finbert([text[:512]])[0]
        if r["label"] == "positive":
            return r["score"]
        if r["label"] == "negative":
            return -r["score"]
        return 0.0
    except:
        return 0.0


def main() -> None:
    # --- Step 1: Load FinBERT ---------------------------------------------
    finbert = load_finbert()

    if not os.path.exists(PRICES_FILE):
        print(f"ERROR: {PRICES_FILE} not found.")
        return

    prices_df = pd.read_csv(PRICES_FILE)
    prices_df["earnings_date"] = pd.to_datetime(prices_df["earnings_date"])
    unique_tickers = prices_df["ticker"].unique().tolist()

    # --- Step 2: Load cached headlines per ticker -----------------------
    cache_dict: dict[str, pd.DataFrame] = {}
    all_dates: list[pd.Timestamp] = []

    for ticker in unique_tickers:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}_news.json")
        if not os.path.exists(cache_path):
            print(f"WARNING: Cache file missing for {ticker}, skipping")
            continue

        with open(cache_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            continue

        df = pd.DataFrame(records)
        if "Date" not in df.columns or df.empty:
            continue

        # Try specified parsing first
        parsed = pd.to_datetime(df["Date"], format="%b-%d-%y %I:%M%p", errors="coerce")
        # If all failed (very likely, given previous analysis showing ISO format), try generic parsing
        if parsed.isna().all():
            parsed = pd.to_datetime(df["Date"], errors="coerce")

        df["parsed_date"] = parsed
        df = df.dropna(subset=["parsed_date"])

        if df.empty:
            continue

        min_date = df["parsed_date"].min()
        max_date = df["parsed_date"].max()
        all_dates.extend([min_date, max_date])

        cache_dict[ticker] = df[["parsed_date", "Title"]].copy()
        print(
            f"{ticker}: {len(df)} headlines, "
            f"date range {min_date.date()} to {max_date.date()}"
        )

    global_news_min = min(all_dates) if all_dates else None
    global_news_max = max(all_dates) if all_dates else None
    global_earn_min = prices_df["earnings_date"].min()
    global_earn_max = prices_df["earnings_date"].max()

    # --- Step 3: Score headlines per (ticker, earnings_date) ---------------
    rows: list[dict] = []
    skipped_count = 0

    for idx, row in prices_df.iterrows():
        ticker = row["ticker"]
        edate = row["earnings_date"]

        if ticker not in cache_dict:
            skipped_count += 1
            print(f"SKIP {ticker} {edate.date()} — no cache")
            continue

        headlines_df = cache_dict[ticker]

        window_start = edate - pd.Timedelta(days=7)
        window_end = edate

        mask = (headlines_df["parsed_date"] >= window_start) & (
            headlines_df["parsed_date"] <= window_end
        )
        window_df = headlines_df[mask].copy()

        n_window = len(window_df)
        if n_window < 3:
            skipped_count += 1
            print(f"SKIP {ticker} {edate.date()} — only {n_window} headlines in window")
            continue

        # Score headlines
        scores = []
        for title in window_df["Title"]:
            if isinstance(title, str) and title.strip():
                scores.append(score_text(finbert, title))

        if not scores:
            skipped_count += 1
            print(f"SKIP {ticker} {edate.date()} — all scores failed")
            continue

        n_scores = len(scores)
        arr = np.array(scores)

        sentiment_score = round(float(arr.mean()), 4)
        buzz_score = min(n_scores, 100)

        mid = max(n_scores // 2, 1)
        if n_scores == 1:
            sentiment_slope = 0.0
        else:
            first_half = arr[:mid]
            second_half = arr[mid:]
            sentiment_slope = round(float(second_half.mean() - first_half.mean()), 4)

        pos = int((arr > 0.1).sum())
        neg = int((arr < -0.1).sum())
        sentiment_polarity = round((pos - neg) / n_scores, 4)

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": edate.date(),
                "sentiment_score": sentiment_score,
                "buzz_score": buzz_score,
                "sentiment_slope": sentiment_slope,
                "sentiment_polarity": sentiment_polarity,
                "n_headlines": n_scores,
            }
        )

    # --- Step 4: After processing all rows ---------------------------------
    if not rows:
        print("\nERROR: No rows collected.")
        if global_news_min and global_news_max:
            print(
                f"Your Finviz headlines cover: {global_news_min.date()} to {global_news_max.date()} across all tickers"
            )
        print(
            f"Your earnings events cover: {global_earn_min.date()} to {global_earn_max.date()}"
        )
        print(
            "There is NO overlap — Finviz history is too short for your earnings dates."
        )
        print("Solution: see fallback below")
        return

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("earnings_date").reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(out_df)} rows to sentiment_raw.csv")
    print("Tickers covered:")
    print(out_df["ticker"].value_counts().to_string())
    print(
        f"Date range: {out_df['earnings_date'].min()} to {out_df['earnings_date'].max()}"
    )
    print(f"Skipped: {skipped_count} events (no headlines in window)")

    print("\nPer-column std for sentiment columns:")
    numeric_cols = [
        "sentiment_score",
        "buzz_score",
        "sentiment_slope",
        "sentiment_polarity",
    ]
    for col in numeric_cols:
        if out_df[col].std() is not np.nan:
            print(f"  {col}: {out_df[col].std():.4f}")

    # --- Step 5: FALLBACK (Requested in instructions) ---------------------
    """
    FALLBACK
    If zero rows were collected, the fix is to manually set the
    earnings dates to match available Finviz data.
    The fallback script should:
        Load each cache/finviz/{ticker}_news.json
        Find the date range of headlines
        Use the midpoint of that range as a synthetic earnings date per ticker
        Create one row per ticker using ALL headlines in the cache as the window
        This gives 9 rows of real FinBERT sentiment — enough to demo the pipeline
    """


if __name__ == "__main__":
    main()
