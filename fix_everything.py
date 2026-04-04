"""
fix_everything.py

Standalone script to rebuild the entire enhanced model pipeline using a
cross-sectional sentiment strategy (one sentiment vector per ticker,
applied to all earnings events for that ticker). No new API calls are made.
"""

import json
import os
import transformers

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join("cache", "finviz")
PRICES_FILE = os.path.join("data", "prices_raw.csv")

BASELINE_OUT = os.path.join("data", "dataset_baseline.csv")
ENHANCED_OUT = os.path.join("data", "dataset_enhanced.csv")
SENTIMENT_OUT = os.path.join("data", "sentiment_raw.csv")
MODELS_DIR = "models"
DATA_DIR = "data"

TICKERS = [
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

BASELINE_FEATURES = ["momentum_7d", "volume_spike", "market_return_7d", "volatility_7d"]
SENTIMENT_FEATURES = [
    "sentiment_score",
    "buzz_score",
    "sentiment_slope",
    "sentiment_polarity",
]
TRANSCRIPT_FEATURES = ["transcript_sentiment"]
ENHANCED_FEATURES = BASELINE_FEATURES + SENTIMENT_FEATURES + TRANSCRIPT_FEATURES

LABEL_COL = "label"

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def load_finbert():
    print("Loading FinBERT...")
    transformers.logging.set_verbosity_error()
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )
    print("FinBERT ready.\n")
    return pipe


def score_text(finbert, text: str) -> float:
    try:
        r = finbert([text[:512]])[0]
        if r["label"] == "positive":
            return r["score"]
        if r["label"] == "negative":
            return -r["score"]
        return 0.0
    except:
        return 0.0


def evaluate_model(X: pd.DataFrame, y: pd.Series, model_name: str) -> pd.DataFrame:
    n_splits = 5
    print(f"--- {model_name} CV ({n_splits} folds) ---")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_test.nunique() < 2:
            print(f"  Fold {fold_idx}: SKIPPED — test fold has only one class")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        y_prob = clf.predict_proba(X_test_s)
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = None

        fold_records.append(
            {
                "fold": fold_idx,
                "accuracy": round(acc, 3),
                "f1": round(f1, 3),
                "roc_auc": round(auc, 3) if auc is not None else None,
            }
        )

        print(f"  Fold {fold_idx}: acc={acc:.3f} f1={f1:.3f} auc={(auc or 0):.3f}")

    results = pd.DataFrame(fold_records)

    if not results.empty:
        print(
            f"  Mean: acc={results['accuracy'].mean():.3f} "
            f"f1={results['f1'].mean():.3f} "
            f"auc={results['roc_auc'].mean():.3f}\n"
        )
    return results


def train_final_model(X, y, model_name, path_prefix):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_s, y)

    joblib.dump(clf, f"{MODELS_DIR}/{path_prefix}.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler_{model_name.lower()[:4]}.pkl")
    return clf, scaler


def main():
    # ------------------------------------------------------------------
    # Step 1 — Load FinBERT
    # ------------------------------------------------------------------
    finbert = load_finbert()

    # ------------------------------------------------------------------
    # Step 2 — Compute one sentiment vector per ticker
    # ------------------------------------------------------------------
    print("=" * 60)
    print("COMPUTING CROSS-SECTIONAL SENTIMENT")
    print("=" * 60)

    ticker_sentiment = {}

    for ticker in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}_news.json")

        if not os.path.exists(cache_path):
            print(f"WARNING: Cache file missing for {ticker}, using zeros")
            ticker_sentiment[ticker] = {
                "ticker": ticker,
                "sentiment_score": 0.0,
                "buzz_score": 100.0,
                "sentiment_slope": 0.0,
                "sentiment_polarity": 0.0,
                "n_headlines": 100,
            }
            continue

        with open(cache_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        scores = []
        for r in records:
            title = r.get("Title", "")
            if title and isinstance(title, str):
                scores.append(score_text(finbert, title))

        if not scores:
            print(f"WARNING: No valid headlines for {ticker}, using zeros")
            ticker_sentiment[ticker] = {
                "ticker": ticker,
                "sentiment_score": 0.0,
                "buzz_score": 100.0,
                "sentiment_slope": 0.0,
                "sentiment_polarity": 0.0,
                "n_headlines": 100,
            }
            continue

        arr = np.array(scores)
        sentiment_score = round(float(arr.mean()), 4)
        buzz_score = 100.0

        mid = 50 if len(arr) >= 100 else max(len(arr) // 2, 1)
        if len(arr) < 2:
            sentiment_slope = 0.0
        else:
            first_half = arr[:mid]
            second_half = arr[mid:]
            sentiment_slope = round(float(second_half.mean() - first_half.mean()), 4)

        pos = int((arr > 0.1).sum())
        neg = int((arr < -0.1).sum())
        sentiment_polarity = round((pos - neg) / 100.0, 4)

        ticker_sentiment[ticker] = {
            "ticker": ticker,
            "sentiment_score": sentiment_score,
            "buzz_score": buzz_score,
            "sentiment_slope": sentiment_slope,
            "sentiment_polarity": sentiment_polarity,
            "n_headlines": 100,
        }

        print(
            f"{ticker}: sentiment={sentiment_score:+.3f} "
            f"polarity={sentiment_polarity:+.3f} slope={sentiment_slope:+.3f}"
        )

    # ------------------------------------------------------------------
    # Step 3 — Verify variance across tickers
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFYING VARIANCE")
    print("=" * 60)

    scores = [v["sentiment_score"] for v in ticker_sentiment.values()]
    print(f"Sentiment scores: {scores}")

    std_val = np.std(scores)
    print(f"Std of sentiment_score across tickers: {std_val:.4f}")
    if std_val < 0.01:
        print("WARNING: sentiment has very low variance across tickers")
    else:
        print("OK: sentiment varies meaningfully across tickers")

    # ------------------------------------------------------------------
    # Step 4 — Build sentiment_raw.csv
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDING SENTIMENT CSV")
    print("=" * 60)

    sent_df = pd.DataFrame(list(ticker_sentiment.values()))
    sent_df.to_csv(SENTIMENT_OUT, index=False)
    print(sent_df.to_string())

    # ------------------------------------------------------------------
    # Step 5 — Build dataset_enhanced.csv
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDING DATASET ENHANCED")
    print("=" * 60)

    if not os.path.exists(PRICES_FILE):
        print(f"ERROR: {PRICES_FILE} missing.")
        return

    prices = pd.read_csv(PRICES_FILE)
    enh_rows = []

    for idx, row in prices.iterrows():
        t = row["ticker"]
        if t in ticker_sentiment:
            row_dict = row.to_dict()
            row_dict.update(ticker_sentiment[t])
            enh_rows.append(row_dict)

    df_enh = pd.DataFrame(enh_rows)
    df_enh["earnings_date"] = pd.to_datetime(df_enh["earnings_date"]).dt.date

    # Merge Transcript Sentiment (LEFT JOIN to not break existing data)
    TRANSCRIPTS_FILE = os.path.join("data", "transcripts_raw.csv")
    if os.path.exists(TRANSCRIPTS_FILE):
        df_trans = pd.read_csv(TRANSCRIPTS_FILE)
        df_trans["earnings_date"] = pd.to_datetime(df_trans["earnings_date"]).dt.date
        df_enh = df_enh.merge(df_trans, on=["ticker", "earnings_date"], how="left")
        # Fill missing transcripts with Neutral (0.0)
        df_enh["transcript_sentiment"] = df_enh["transcript_sentiment"].fillna(0.0)
        print(f"Merged transcripts: {len(df_trans)} rows available")
    else:
        df_enh["transcript_sentiment"] = 0.0
        print(f"WARNING: {TRANSCRIPTS_FILE} missing, zeroing transcript sentiment")

    df_enh = df_enh.sort_values("earnings_date").reset_index(drop=True)

    # Clean NaNs and infs
    check_cols = (
        BASELINE_FEATURES + SENTIMENT_FEATURES + TRANSCRIPT_FEATURES + [LABEL_COL]
    )
    df_enh[check_cols] = df_enh[check_cols].replace([np.inf, -np.inf], np.nan)
    df_enh = df_enh.dropna(subset=check_cols)

    print(f"Shape: {df_enh.shape}")
    dist = df_enh[LABEL_COL].value_counts().sort_index()
    print(f"Label distribution:\n{dist.to_string()}")

    print("\nPer-column std for sentiment columns:")
    for col in SENTIMENT_FEATURES + TRANSCRIPT_FEATURES:
        print(f"  {col}: {df_enh[col].std():.4f}")

    df_enh.to_csv(ENHANCED_OUT, index=False)

    df_base = pd.read_csv(BASELINE_OUT)
    df_base["earnings_date"] = pd.to_datetime(df_base["earnings_date"]).dt.date
    df_base = df_base.sort_values("earnings_date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 6 — Train and evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING MODELS")
    print("=" * 60)

    res_base = evaluate_model(
        df_base[BASELINE_FEATURES], df_base[LABEL_COL], "Baseline"
    )
    res_enh = evaluate_model(df_enh[ENHANCED_FEATURES], df_enh[LABEL_COL], "Enhanced")

    # ------------------------------------------------------------------
    # Step 7 — Train final models
    # ------------------------------------------------------------------
    clf_base, _ = train_final_model(
        df_base[BASELINE_FEATURES], df_base[LABEL_COL], "Baseline", "baseline"
    )
    clf_enh, scaler_enh = train_final_model(
        df_enh[ENHANCED_FEATURES], df_enh[LABEL_COL], "Enhanced", "enhanced"
    )

    # ------------------------------------------------------------------
    # Step 8 — Add predictions to dataset_enhanced.csv
    # ------------------------------------------------------------------
    X_enh_full_s = scaler_enh.transform(df_enh[ENHANCED_FEATURES])
    df_enh["predicted"] = clf_enh.predict(X_enh_full_s)
    df_enh["actual_return"] = df_enh["5d_return"].round(4)
    df_enh["correct"] = (df_enh["predicted"] == df_enh[LABEL_COL]).astype(int)
    df_enh.to_csv(ENHANCED_OUT, index=False)

    # ------------------------------------------------------------------
    # Step 9 — Save dashboard JSONs
    # ------------------------------------------------------------------
    comp = {
        "baseline": {
            "accuracy": round(float(res_base["accuracy"].mean()), 3),
            "f1": round(float(res_base["f1"].mean()), 3),
            "auc": round(float(res_base["roc_auc"].mean()), 3),
            "n_rows": len(df_base),
            "n_tickers": int(df_base["ticker"].nunique()),
        },
        "enhanced": {
            "accuracy": round(float(res_enh["accuracy"].mean()), 3),
            "f1": round(float(res_enh["f1"].mean()), 3),
            "auc": round(float(res_enh["roc_auc"].mean()), 3),
            "n_rows": len(df_enh),
            "n_tickers": int(df_enh["ticker"].nunique()),
        },
    }
    with open(os.path.join(DATA_DIR, "model_comparison.json"), "w") as f:
        json.dump(comp, f, indent=2)

    fi_base = [
        {"feature": f, "importance": round(float(i), 3)}
        for f, i in zip(BASELINE_FEATURES, clf_base.feature_importances_)
    ]
    fi_base.sort(key=lambda x: x["importance"], reverse=True)
    with open(os.path.join(DATA_DIR, "feature_importance_baseline.json"), "w") as f:
        json.dump(fi_base, f, indent=2)

    fi_enh = [
        {"feature": f, "importance": round(float(i), 3)}
        for f, i in zip(ENHANCED_FEATURES, clf_enh.feature_importances_)
    ]
    fi_enh.sort(key=lambda x: x["importance"], reverse=True)
    with open(os.path.join(DATA_DIR, "feature_importance_enhanced.json"), "w") as f:
        json.dump(fi_enh, f, indent=2)

    # Merge fold results
    max_folds = max(len(res_base), len(res_enh))
    fold_out = []

    for f_idx in range(1, max_folds + 1):
        d = {"fold": f_idx}

        rb = res_base[res_base["fold"] == f_idx]
        if not rb.empty and pd.notna(rb["accuracy"].iloc[0]):
            d["baseline_accuracy"] = round(float(rb["accuracy"].iloc[0]), 3)
        else:
            d["baseline_accuracy"] = None

        re = res_enh[res_enh["fold"] == f_idx]
        if not re.empty and pd.notna(re["accuracy"].iloc[0]):
            d["enhanced_accuracy"] = round(float(re["accuracy"].iloc[0]), 3)
        else:
            d["enhanced_accuracy"] = None

        fold_out.append(d)

    with open(os.path.join(DATA_DIR, "fold_results.json"), "w") as f:
        json.dump(fold_out, f, indent=2)

    # ------------------------------------------------------------------
    # Step 10 — Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== FINAL RESULTS ===")
    print("=" * 60)
    print(
        f"Baseline  ({comp['baseline']['n_rows']} rows, {comp['baseline']['n_tickers']} tickers): "
        f"acc={comp['baseline']['accuracy']} "
        f"f1={comp['baseline']['f1']} auc={comp['baseline']['auc']}"
    )
    print(
        f"Enhanced  ({comp['enhanced']['n_rows']} rows, {comp['enhanced']['n_tickers']} tickers): "
        f"acc={comp['enhanced']['accuracy']} "
        f"f1={comp['enhanced']['f1']} auc={comp['enhanced']['auc']}"
    )

    delta_acc = round(comp["enhanced"]["accuracy"] - comp["baseline"]["accuracy"], 3)
    delta_f1 = round(comp["enhanced"]["f1"] - comp["baseline"]["f1"], 3)
    delta_auc = round(comp["enhanced"]["auc"] - comp["baseline"]["auc"], 3)

    print(
        f"Delta: acc={'+' if delta_acc > 0 else ''}{delta_acc} "
        f"f1={'+' if delta_f1 > 0 else ''}{delta_f1} "
        f"auc={'+' if delta_auc > 0 else ''}{delta_auc}"
    )

    print("Top 5 features (enhanced):")
    for row in fi_enh[:5]:
        print(f"  {row['feature']}: {row['importance']}")

    print("All files saved.")


if __name__ == "__main__":
    main()
