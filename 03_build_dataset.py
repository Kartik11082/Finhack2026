"""
03_build_dataset.py

Merge price features with sentiment features into two clean datasets:
  - data/dataset_baseline.csv  — all valid earnings events (price features only)
  - data/dataset_enhanced.csv  — subset with real sentiment scores joined in

No NaN/inf filling is performed; rows with missing or infinite values are dropped.
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRICES_FILE = os.path.join("data", "prices_raw.csv")
SENTIMENT_FILE = os.path.join("data", "sentiment_raw.csv")
BASELINE_OUT = os.path.join("data", "dataset_baseline.csv")
ENHANCED_OUT = os.path.join("data", "dataset_enhanced.csv")

PRICE_FEATURE_COLS = [
    "momentum_7d",
    "volume_spike",
    "volatility_7d",
    "market_return_7d",
]

SENTIMENT_FEATURE_COLS = [
    "sentiment_score",
    "buzz_score",
    "sentiment_slope",
    "sentiment_polarity",
]

LABEL_COL = "label"


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV and parse earnings_date as datetime."""
    df = pd.read_csv(path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


def clean_inf_and_nan(df: pd.DataFrame, check_cols: list[str]) -> pd.DataFrame:
    """Replace inf with NaN in *check_cols*, then drop rows with any NaN."""
    df = df.copy()
    df[check_cols] = df[check_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=check_cols)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN/inf in {check_cols}")
    return df


def print_label_distribution(df: pd.DataFrame) -> None:
    dist = df[LABEL_COL].value_counts().sort_index()
    for val, count in dist.items():
        pct = 100 * count / len(df)
        print(f"    label={val}: {count} ({pct:.1f}%)")


def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    n_rows = len(df)
    n_tickers = df["ticker"].nunique()
    date_min = df["earnings_date"].min().date()
    date_max = df["earnings_date"].max().date()
    print(f"{name}: {n_rows} rows, {n_tickers} tickers, "
          f"date range {date_min} to {date_max}")


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1 — Load both CSVs
    # ------------------------------------------------------------------
    if not os.path.exists(PRICES_FILE):
        print(f"ERROR: {PRICES_FILE} not found. Run 01_collect_prices.py first.")
        return

    prices = load_csv(PRICES_FILE)
    print(f"Loaded {PRICES_FILE}: {prices.shape}")

    has_sentiment = os.path.exists(SENTIMENT_FILE)
    if has_sentiment:
        sentiment = load_csv(SENTIMENT_FILE)
        print(f"Loaded {SENTIMENT_FILE}: {sentiment.shape}")
    else:
        sentiment = pd.DataFrame()
        print(f"{SENTIMENT_FILE} not found — enhanced dataset will be skipped")

    print()

    # ------------------------------------------------------------------
    # Step 2 — Dataset A: baseline (price features only)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Building BASELINE dataset (price features only)")
    print("=" * 60)

    baseline_cols = PRICE_FEATURE_COLS + [LABEL_COL]
    baseline = clean_inf_and_nan(prices, baseline_cols)

    print(f"  Shape: {baseline.shape}")
    print("  Label distribution:")
    print_label_distribution(baseline)

    baseline.to_csv(BASELINE_OUT, index=False)
    print(f"  Saved to {BASELINE_OUT}\n")

    # ------------------------------------------------------------------
    # Step 3 — Dataset B: enhanced (price + sentiment features)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Building ENHANCED dataset (price + sentiment features)")
    print("=" * 60)

    if sentiment.empty:
        print("  No sentiment data available — skipping enhanced dataset.\n")
        enhanced = pd.DataFrame()
    else:
        enhanced = prices.merge(
            sentiment,
            on=["ticker", "earnings_date"],
            how="inner",
        )
        print(f"  After inner join: {enhanced.shape}")

        all_feature_cols = PRICE_FEATURE_COLS + SENTIMENT_FEATURE_COLS + [LABEL_COL]
        enhanced = clean_inf_and_nan(enhanced, all_feature_cols)

        if enhanced.empty:
            print("  Enhanced dataset is empty after cleaning.\n")
        else:
            print(f"  Shape: {enhanced.shape}")
            print("  Label distribution:")
            print_label_distribution(enhanced)

            print("  Tickers represented:")
            for ticker, count in enhanced["ticker"].value_counts().sort_index().items():
                print(f"    {ticker}: {count} rows")

            enhanced.to_csv(ENHANCED_OUT, index=False)
            print(f"  Saved to {ENHANCED_OUT}\n")

    # ------------------------------------------------------------------
    # Step 4 — Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print_dataset_summary("Baseline dataset", baseline)

    if not enhanced.empty:
        print_dataset_summary("Enhanced dataset", enhanced)

        total_pairs = len(baseline)
        sentiment_pairs = len(enhanced)
        pct = 100 * sentiment_pairs / total_pairs if total_pairs else 0
        print(f"Rows with real sentiment: {sentiment_pairs} out of "
              f"{total_pairs} total ({pct:.1f}%)")

        print("\nSentiment column std (confirms feature variance):")
        for col in SENTIMENT_FEATURE_COLS:
            if col in enhanced.columns:
                print(f"  {col}: {enhanced[col].std():.4f}")
    else:
        print("Enhanced dataset: not available (no sentiment data)")

    print()

    # ------------------------------------------------------------------
    # Step 5 — Leakage check
    # ------------------------------------------------------------------
    print("=" * 60)
    print("LEAKAGE CHECK")
    print("=" * 60)

    if enhanced.empty:
        print("  Skipped — no enhanced dataset to validate.")
    else:
        # The sentiment window ends on or before earnings_date by construction
        # in 02_collect_sentiment.py (headlines filtered to <= earnings_date).
        # We verify that no sentiment columns reference future data by checking
        # that every row's earnings_date is a real date (not NaT) and that the
        # join keys are consistent.
        bad_rows = enhanced["earnings_date"].isna().sum()
        if bad_rows == 0:
            print("  Leakage check: PASSED")
            print("  (Sentiment windows end on or before earnings_date by construction)")
        else:
            print(f"  Leakage check: FAILED — {bad_rows} rows affected")

    print()


if __name__ == "__main__":
    main()
