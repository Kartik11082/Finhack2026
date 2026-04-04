"""
04_train_models.py

Train baseline and enhanced RandomForest classifiers with time-aware
cross-validation (TimeSeriesSplit), save models, and generate all JSON
output files for the dashboard.

Outputs:
  models/baseline.pkl, models/scaler_base.pkl
  models/enhanced.pkl, models/scaler_enh.pkl   (only if enhanced data exists)
  data/model_comparison.json
  data/feature_importance_baseline.json
  data/feature_importance_enhanced.json
  data/fold_results.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASELINE_FILE = os.path.join("data", "dataset_baseline.csv")
ENHANCED_FILE = os.path.join("data", "dataset_enhanced.csv")
MODELS_DIR = "models"
DATA_DIR = "data"

BASELINE_FEATURES: list[str] = [
    "momentum_7d", "volume_spike", "market_return_7d", "volatility_7d",
]
SENTIMENT_FEATURES: list[str] = [
    "sentiment_score", "buzz_score", "sentiment_slope", "sentiment_polarity",
]
ENHANCED_FEATURES: list[str] = BASELINE_FEATURES + SENTIMENT_FEATURES

LABEL_COL = "label"


def make_classifier() -> RandomForestClassifier:
    """Return a fresh RandomForestClassifier with identical settings."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )


def choose_n_splits(n_rows: int) -> int:
    return 3 if n_rows < 60 else 5


def print_dataset_info(name: str, df: pd.DataFrame) -> None:
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['earnings_date'].min().date()} to "
          f"{df['earnings_date'].max().date()}")
    print(f"  Unique tickers: {df['ticker'].nunique()} "
          f"({', '.join(sorted(df['ticker'].unique()))})")
    dist = df[LABEL_COL].value_counts().sort_index()
    for val, count in dist.items():
        print(f"    label={val}: {count} ({100 * count / len(df):.1f}%)")


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int,
) -> pd.DataFrame:
    """
    Time-aware cross-validation. Returns a DataFrame with per-fold metrics.
    """
    print(f"\n--- {model_name} CV ({n_splits} folds) ---")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Check for single-class test fold.
        if y_test.nunique() < 2:
            print(f"  Fold {fold_idx}: SKIPPED — test fold has only one class")
            fold_records.append({
                "fold": fold_idx,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "accuracy": None,
                "f1": None,
                "roc_auc": None,
            })
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = make_classifier()
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # roc_auc needs probability estimates.
        y_prob = clf.predict_proba(X_test_s)
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = None

        fold_records.append({
            "fold": fold_idx,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "accuracy": round(acc, 3),
            "f1": round(f1, 3),
            "roc_auc": round(auc, 3) if auc is not None else None,
        })

        print(f"  Fold {fold_idx}: train={len(y_train)} test={len(y_test)}  "
              f"acc={acc:.3f}  f1={f1:.3f}  "
              f"auc={auc:.3f}" if auc is not None else
              f"  Fold {fold_idx}: train={len(y_train)} test={len(y_test)}  "
              f"acc={acc:.3f}  f1={f1:.3f}  auc=N/A")

    results = pd.DataFrame(fold_records)

    # Mean of valid folds only.
    valid = results.dropna(subset=["accuracy"])
    if not valid.empty:
        mean_acc = valid["accuracy"].mean()
        mean_f1 = valid["f1"].mean()
        mean_auc = valid["roc_auc"].mean() if valid["roc_auc"].notna().any() else None
        print(f"  Mean: acc={mean_acc:.3f}  f1={mean_f1:.3f}  "
              f"auc={mean_auc:.3f}" if mean_auc is not None else
              f"  Mean: acc={mean_acc:.3f}  f1={mean_f1:.3f}  auc=N/A")
    else:
        print("  No valid folds to average!")

    return results


def get_feature_importance(clf: RandomForestClassifier, features: list[str]) -> list[dict]:
    """Return sorted list of {feature, importance} dicts."""
    importances = clf.feature_importances_
    pairs = [
        {"feature": f, "importance": round(float(imp), 3)}
        for f, imp in zip(features, importances)
    ]
    pairs.sort(key=lambda x: x["importance"], reverse=True)
    return pairs


def mean_metric(results: pd.DataFrame, col: str) -> float | None:
    """Compute mean of a metric column from valid folds."""
    valid = results[col].dropna()
    if valid.empty:
        return None
    return round(float(valid.mean()), 3)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load datasets
    # ------------------------------------------------------------------
    if not os.path.exists(BASELINE_FILE):
        print(f"ERROR: {BASELINE_FILE} not found. Run 03_build_dataset.py first.")
        return

    df_base = pd.read_csv(BASELINE_FILE)
    df_base["earnings_date"] = pd.to_datetime(df_base["earnings_date"])
    df_base = df_base.sort_values("earnings_date").reset_index(drop=True)

    has_enhanced = os.path.exists(ENHANCED_FILE)
    if has_enhanced:
        df_enh = pd.read_csv(ENHANCED_FILE)
        df_enh["earnings_date"] = pd.to_datetime(df_enh["earnings_date"])
        df_enh = df_enh.sort_values("earnings_date").reset_index(drop=True)
        if df_enh.empty:
            has_enhanced = False
            print(f"{ENHANCED_FILE} is empty — enhanced model will be skipped.\n")
    else:
        df_enh = pd.DataFrame()
        print(f"{ENHANCED_FILE} not found — enhanced model will be skipped.\n")

    # ------------------------------------------------------------------
    # Step 2 — Print dataset info
    # ------------------------------------------------------------------
    print("=" * 60)
    print("BASELINE DATASET")
    print("=" * 60)
    print_dataset_info("Baseline", df_base)

    if has_enhanced:
        print("\n" + "=" * 60)
        print("ENHANCED DATASET")
        print("=" * 60)
        print_dataset_info("Enhanced", df_enh)

    # ------------------------------------------------------------------
    # Step 3 — Choose n_splits
    # ------------------------------------------------------------------
    n_splits_base = choose_n_splits(len(df_base))
    print(f"\nBaseline n_splits: {n_splits_base} (N={len(df_base)})")

    n_splits_enh = None
    if has_enhanced:
        n_splits_enh = choose_n_splits(len(df_enh))
        print(f"Enhanced n_splits: {n_splits_enh} (N={len(df_enh)})")

    # ------------------------------------------------------------------
    # Step 4 + 5 — Evaluate models
    # ------------------------------------------------------------------
    X_base = df_base[BASELINE_FEATURES]
    y_base = df_base[LABEL_COL]
    baseline_results = evaluate_model(X_base, y_base, "Baseline", n_splits_base)

    enhanced_results = pd.DataFrame()
    if has_enhanced:
        X_enh = df_enh[ENHANCED_FEATURES]
        y_enh = df_enh[LABEL_COL]
        enhanced_results = evaluate_model(X_enh, y_enh, "Enhanced", n_splits_enh)

    # ------------------------------------------------------------------
    # Step 6 — Train final models on full data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODELS")
    print("=" * 60)

    # Baseline
    scaler_base = StandardScaler()
    X_base_s = scaler_base.fit_transform(X_base)
    clf_base = make_classifier()
    clf_base.fit(X_base_s, y_base)
    joblib.dump(clf_base, os.path.join(MODELS_DIR, "baseline.pkl"))
    joblib.dump(scaler_base, os.path.join(MODELS_DIR, "scaler_base.pkl"))
    print("  Saved models/baseline.pkl, models/scaler_base.pkl")

    clf_enh = None
    scaler_enh = None
    if has_enhanced:
        scaler_enh = StandardScaler()
        X_enh_s = scaler_enh.fit_transform(X_enh)
        clf_enh = make_classifier()
        clf_enh.fit(X_enh_s, y_enh)
        joblib.dump(clf_enh, os.path.join(MODELS_DIR, "enhanced.pkl"))
        joblib.dump(scaler_enh, os.path.join(MODELS_DIR, "scaler_enh.pkl"))
        print("  Saved models/enhanced.pkl, models/scaler_enh.pkl")

    # ------------------------------------------------------------------
    # Step 7 — Add predictions to enhanced dataset (audit log)
    # ------------------------------------------------------------------
    if has_enhanced and clf_enh is not None and scaler_enh is not None:
        X_enh_full_s = scaler_enh.transform(df_enh[ENHANCED_FEATURES])
        df_enh["predicted"] = clf_enh.predict(X_enh_full_s)
        df_enh["actual_return"] = df_enh["5d_return"].round(4)
        df_enh["correct"] = (df_enh["predicted"] == df_enh[LABEL_COL]).astype(int)
        df_enh.to_csv(ENHANCED_FILE, index=False)
        print(f"  Updated {ENHANCED_FILE} with predictions")

    # ------------------------------------------------------------------
    # Step 8 — Save dashboard JSON files
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING DASHBOARD FILES")
    print("=" * 60)

    # model_comparison.json
    comparison = {
        "baseline": {
            "accuracy": mean_metric(baseline_results, "accuracy"),
            "f1": mean_metric(baseline_results, "f1"),
            "auc": mean_metric(baseline_results, "roc_auc"),
            "n_rows": len(df_base),
            "n_tickers": int(df_base["ticker"].nunique()),
        },
        "enhanced": {
            "accuracy": mean_metric(enhanced_results, "accuracy") if has_enhanced else None,
            "f1": mean_metric(enhanced_results, "f1") if has_enhanced else None,
            "auc": mean_metric(enhanced_results, "roc_auc") if has_enhanced else None,
            "n_rows": len(df_enh) if has_enhanced else 0,
            "n_tickers": int(df_enh["ticker"].nunique()) if has_enhanced else 0,
        },
    }
    with open(os.path.join(DATA_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print("  Saved data/model_comparison.json")

    # feature_importance_baseline.json
    fi_base = get_feature_importance(clf_base, BASELINE_FEATURES)
    with open(os.path.join(DATA_DIR, "feature_importance_baseline.json"), "w") as f:
        json.dump(fi_base, f, indent=2)
    print("  Saved data/feature_importance_baseline.json")

    # feature_importance_enhanced.json
    if has_enhanced and clf_enh is not None:
        fi_enh = get_feature_importance(clf_enh, ENHANCED_FEATURES)
    else:
        fi_enh = []
    with open(os.path.join(DATA_DIR, "feature_importance_enhanced.json"), "w") as f:
        json.dump(fi_enh, f, indent=2)
    print("  Saved data/feature_importance_enhanced.json")

    # fold_results.json — align baseline and enhanced by fold number
    max_folds = max(
        len(baseline_results),
        len(enhanced_results) if has_enhanced else 0,
    )
    fold_results: list[dict] = []
    for fold_num in range(1, max_folds + 1):
        entry: dict = {"fold": fold_num}

        base_row = baseline_results[baseline_results["fold"] == fold_num]
        entry["baseline_accuracy"] = (
            float(base_row["accuracy"].iloc[0])
            if not base_row.empty and base_row["accuracy"].notna().iloc[0]
            else None
        )

        if has_enhanced and not enhanced_results.empty:
            enh_row = enhanced_results[enhanced_results["fold"] == fold_num]
            entry["enhanced_accuracy"] = (
                float(enh_row["accuracy"].iloc[0])
                if not enh_row.empty and enh_row["accuracy"].notna().iloc[0]
                else None
            )
        else:
            entry["enhanced_accuracy"] = None

        fold_results.append(entry)

    with open(os.path.join(DATA_DIR, "fold_results.json"), "w") as f:
        json.dump(fold_results, f, indent=2)
    print("  Saved data/fold_results.json")

    # ------------------------------------------------------------------
    # Step 9 — Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== FINAL RESULTS ===")
    print("=" * 60)

    base_acc = comparison["baseline"]["accuracy"]
    base_f1 = comparison["baseline"]["f1"]
    base_auc = comparison["baseline"]["auc"]
    print(f"Baseline  ({comparison['baseline']['n_rows']} rows, "
          f"{comparison['baseline']['n_tickers']} tickers): "
          f"acc={base_acc}  f1={base_f1}  auc={base_auc}")

    enh_acc = comparison["enhanced"]["accuracy"]
    enh_f1 = comparison["enhanced"]["f1"]
    enh_auc = comparison["enhanced"]["auc"]

    if has_enhanced and enh_acc is not None:
        print(f"Enhanced  ({comparison['enhanced']['n_rows']} rows, "
              f"{comparison['enhanced']['n_tickers']} tickers): "
              f"acc={enh_acc}  f1={enh_f1}  auc={enh_auc}")

        delta_acc = round(enh_acc - base_acc, 3) if base_acc is not None else None
        delta_f1 = round(enh_f1 - base_f1, 3) if base_f1 is not None else None
        delta_auc = round(enh_auc - base_auc, 3) if base_auc is not None else None
        print(f"Delta:    acc={'+' if delta_acc and delta_acc >= 0 else ''}{delta_acc}  "
              f"f1={'+' if delta_f1 and delta_f1 >= 0 else ''}{delta_f1}  "
              f"auc={'+' if delta_auc and delta_auc >= 0 else ''}{delta_auc}")
    else:
        print("Enhanced  — not available (no sentiment data)")

    print("\nTop features (baseline):")
    for entry in fi_base[:5]:
        print(f"  {entry['feature']}: {entry['importance']}")

    if fi_enh:
        print("\nTop features (enhanced):")
        for entry in fi_enh[:5]:
            print(f"  {entry['feature']}: {entry['importance']}")

    print("\nAll files saved to data/ and models/")


if __name__ == "__main__":
    main()
