"""
01_collect_prices.py

Fetch stock prices and earnings dates for a universe of tickers using yfinance,
compute price features around each earnings event, and save the result as
data/prices_raw.csv.

Features computed per earnings event:
  - momentum_7d: 7-trading-day price momentum ending on earnings date
  - volume_spike: recent 3-day avg volume vs. longer-term avg volume
  - volatility_7d: std of daily returns over 7 trading days before earnings
  - market_return_7d: SPY 7-trading-day return ending on earnings date
  - 5d_return: 5-trading-day forward return after earnings
  - label: 1 if price is higher 5 trading days after earnings, else 0
"""

import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKERS: list[str] = [
    "XOM",
    "CVX",
    "2222.SR",
    "LNG",
    "SHEL",
    "KEX",
    "FRO",
    "CF",
    "NTR",
    "ADM",
    "DE",
    "CAT",
    "HON",
    "NEE",
    "DUK",
    "NESN.SW",
    "PEP",
]

DATE_START = pd.Timestamp("2020-01-01")
DATE_END = pd.Timestamp("2026-04-01")
TODAY = pd.Timestamp(date.today())

CALENDAR_DAYS_BEFORE = 40
CALENDAR_DAYS_AFTER = 10

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "prices_raw.csv")


def download_spy(start: str, end: str) -> pd.DataFrame:
    """Download SPY prices once for the full date range needed."""
    print("Downloading SPY data for market return baseline...")
    spy = yf.download("SPY", start=start, end=end, progress=False)

    # yf.download may return MultiIndex columns when given a single ticker;
    # flatten if needed.
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    return spy


def get_earnings_dates(ticker: str) -> list[date]:
    """Return filtered, sorted list of past earnings dates for *ticker*."""
    ticker_obj = yf.Ticker(ticker)
    earnings = ticker_obj.earnings_dates

    if earnings is None or earnings.empty:
        return []

    # Convert index to tz-naive timestamps, then keep only the date part.
    idx = earnings.index.tz_localize(None) if earnings.index.tz else earnings.index
    dates = idx.normalize()  # midnight timestamps

    # Filter to the allowed window and only past dates.
    mask = (dates >= DATE_START) & (dates < DATE_END) & (dates < TODAY)
    filtered = dates[mask].unique()

    return sorted([d.date() for d in filtered])


def snap_to_nearest_trading_day(
    price_index: pd.DatetimeIndex, target: date
) -> int | None:
    """
    Find *target* in *price_index*. If the exact date is missing, snap to the
    nearest trading day. Returns the integer position, or None if nothing is
    close enough (within 5 calendar days).
    """
    target_ts = pd.Timestamp(target)
    if target_ts in price_index:
        return price_index.get_loc(target_ts)

    # Find nearest date in the index.
    diffs = abs(price_index - target_ts)
    nearest_idx = diffs.argmin()
    if diffs[nearest_idx] > pd.Timedelta(days=5):
        return None
    return nearest_idx


def compute_row(
    ticker: str,
    earnings_date: date,
    prices: pd.DataFrame,
    spy_prices: pd.DataFrame,
) -> dict | None:
    """
    Compute features and label for a single earnings event.
    Returns a dict of column values, or None if the row should be skipped.
    """
    close = prices["Close"]
    volume = prices["Volume"]

    # --- Snap earnings date to nearest trading day position ----------------
    pos_e = snap_to_nearest_trading_day(prices.index, earnings_date)
    if pos_e is None:
        print(
            f"  WARNING: skipping {ticker} {earnings_date} — "
            "could not snap to a trading day"
        )
        return None

    # --- Basic data-sufficiency check -------------------------------------
    if len(prices) < 10:
        print(
            f"  WARNING: skipping {ticker} {earnings_date} — "
            f"only {len(prices)} trading days in window (need >= 10)"
        )
        return None

    # --- Label: 5-trading-day forward return ------------------------------
    if pos_e + 5 >= len(close):
        print(
            f"  WARNING: skipping {ticker} {earnings_date} — "
            "not enough forward data for 5d label"
        )
        return None

    close_e = close.iloc[pos_e]
    close_e5 = close.iloc[pos_e + 5]
    five_day_return = (close_e5 - close_e) / close_e
    label = 1 if close_e5 > close_e else 0

    # --- momentum_7d ------------------------------------------------------
    if pos_e - 7 < 0:
        print(
            f"  WARNING: skipping {ticker} {earnings_date} — "
            "not enough lookback for momentum_7d"
        )
        return None
    close_7 = close.iloc[pos_e - 7]
    momentum_7d = (close_e - close_7) / close_7

    # --- volume_spike -----------------------------------------------------
    if pos_e - 3 < 0:
        print(
            f"  WARNING: skipping {ticker} {earnings_date} — "
            "not enough data for volume_spike numerator"
        )
        return None
    recent_vol = volume.iloc[pos_e - 3 : pos_e].mean()

    long_start = max(pos_e - 30, 0)
    long_end = max(pos_e - 7, 0)
    if long_end <= long_start:
        volume_spike = 1.0
    else:
        long_vol = volume.iloc[long_start:long_end].mean()
        volume_spike = (recent_vol / long_vol) if long_vol != 0 else 1.0

    # --- volatility_7d ----------------------------------------------------
    returns_window = close.iloc[max(pos_e - 7, 0) : pos_e + 1].pct_change().dropna()
    volatility_7d = returns_window.std() if len(returns_window) > 0 else np.nan

    # --- market_return_7d (SPY) -------------------------------------------
    spy_close = spy_prices["Close"]
    spy_pos = snap_to_nearest_trading_day(spy_prices.index, earnings_date)
    if spy_pos is None or spy_pos - 7 < 0:
        print(
            f"  WARNING: skipping {ticker} {earnings_date} — "
            "could not compute market_return_7d from SPY"
        )
        return None
    spy_e = spy_close.iloc[spy_pos]
    spy_7 = spy_close.iloc[spy_pos - 7]
    market_return_7d = (spy_e - spy_7) / spy_7

    # --- Final validity check ---------------------------------------------
    values = {
        "ticker": ticker,
        "earnings_date": earnings_date,
        "momentum_7d": momentum_7d,
        "volume_spike": volume_spike,
        "volatility_7d": volatility_7d,
        "market_return_7d": market_return_7d,
        "5d_return": five_day_return,
        "label": label,
    }

    numeric_vals = [
        momentum_7d,
        volume_spike,
        volatility_7d,
        market_return_7d,
        five_day_return,
    ]
    for v in numeric_vals:
        if np.isnan(v) or np.isinf(v):
            print(
                f"  WARNING: skipping {ticker} {earnings_date} — "
                "feature contains NaN or inf"
            )
            return None

    return values


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Download SPY once for the entire range -------------------------
    spy_start = (DATE_START.date() - timedelta(days=CALENDAR_DAYS_BEFORE)).isoformat()
    spy_end = (TODAY.date() + timedelta(days=CALENDAR_DAYS_AFTER)).isoformat()
    spy_prices = download_spy(spy_start, spy_end)

    if spy_prices.empty:
        print("ERROR: failed to download SPY data. Exiting.")
        return

    print(
        f"SPY data: {len(spy_prices)} trading days "
        f"({spy_prices.index.min().date()} to {spy_prices.index.max().date()})\n"
    )

    rows: list[dict] = []

    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"Fetching earnings dates for {ticker}...")

        try:
            earnings_dates = get_earnings_dates(ticker)
        except Exception as exc:
            print(f"  ERROR getting earnings dates for {ticker}: {exc}")
            continue

        if not earnings_dates:
            print(f"  No earnings dates found for {ticker}")
            continue

        print(f"  Found {len(earnings_dates)} earnings events")

        for edate in earnings_dates:
            print(f"  Processing {ticker} {edate}...")
            try:
                dl_start = (edate - timedelta(days=CALENDAR_DAYS_BEFORE)).isoformat()
                dl_end = (edate + timedelta(days=CALENDAR_DAYS_AFTER)).isoformat()

                prices = yf.download(ticker, start=dl_start, end=dl_end, progress=False)

                if isinstance(prices.columns, pd.MultiIndex):
                    prices.columns = prices.columns.get_level_values(0)

                if prices.empty:
                    print(f"    WARNING: no price data returned, skipping")
                    continue

                row = compute_row(ticker, edate, prices, spy_prices)
                if row is not None:
                    rows.append(row)

            except Exception as exc:
                print(f"    ERROR processing {ticker} {edate}: {exc}")
                continue

        # Rate-limit between tickers.
        time.sleep(1)

    # ----- Build and save output ------------------------------------------
    if not rows:
        print("\nNo valid rows collected. Nothing to save.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("earnings_date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
