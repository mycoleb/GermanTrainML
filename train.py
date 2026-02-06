from utils import to_float32

# from __future__ import annotations

from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

import sys
from pathlib import Path
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.linear_model import SGDClassifier, SGDRegressor

DATA_PATH = Path("data/processed/dataset.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS_EXPECTED = ["station_id", "category", "train_id"]
NUM_COLS_EXPECTED = ["hour", "dow", "month", "is_weekend", "station_delay_mean_20"]


def main() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/processed/dataset.parquet. Run p.py first.")

    print(f"Loading: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # Targets
    y_cls = df["is_delayed_5"].astype(int)
    y_reg = pd.to_numeric(df["delay_minutes"], errors="coerce").astype("float32")

    # Build X
    feature_cols = [c for c in df.columns if c not in ("is_delayed_5", "delay_minutes", "scheduled_time")]
    X = df[feature_cols].copy()

    # --- HARD FIX: remove pandas nullable dtypes + pandas.NA BEFORE sklearn sees it ---

    # Categorical -> plain python strings (object) and fill missing
    # Categorical cleanup
    for c in CAT_COLS_EXPECTED:
        if c in X.columns:
            X[c] = (
                X[c].astype("string")
                    .fillna("UNKNOWN")
                    .replace({"": "UNKNOWN"})
                    .astype(str)
            )

    # Numeric cleanup
    for c in NUM_COLS_EXPECTED:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32").fillna(-1.0)

    # Explicit columns
    cat_cols = [c for c in CAT_COLS_EXPECTED if c in X.columns]
    num_cols = [c for c in NUM_COLS_EXPECTED if c in X.columns]

    print("Features:")
    print("  categorical:", cat_cols)
    print("  numeric    :", num_cols)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("to_float", FunctionTransformer(to_float32, feature_names_out="one-to-one")),
    ])
    
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )


    

    # Sparse-friendly + scales to millions of rows
    cls = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=30,
        tol=1e-3,
        early_stopping=True,
        n_iter_no_change=3,
        random_state=42,
    )

    reg = SGDRegressor(
        loss="squared_error",
        alpha=1e-5,
        max_iter=30,
        tol=1e-3,
        early_stopping=True,
        n_iter_no_change=3,
        random_state=42,
    )

    cls_pipe = Pipeline([("pre", pre), ("model", cls)])
    reg_pipe = Pipeline([("pre", pre), ("model", reg)])

    # Split
    X_train, X_test, ycls_train, ycls_test, yreg_train, yreg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    print("Training classifier...")
    cls_pipe.fit(X_train, ycls_train)

    print("Training regressor...")
    reg_pipe.fit(X_train, yreg_train)

    print("Evaluating...")
    proba = cls_pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(ycls_test, proba)
    ap = average_precision_score(ycls_test, proba)

    pred_min = reg_pipe.predict(X_test)
    mae = mean_absolute_error(yreg_test, pred_min)

    print("\n=== METRICS ===")
    print(f"Classifier AUC: {auc:.4f}")
    print(f"Classifier AP : {ap:.4f}")
    print(f"Regressor  MAE: {mae:.3f} minutes")

    dump(cls_pipe, MODEL_DIR / "delay_classifier.joblib")
    dump(reg_pipe, MODEL_DIR / "delay_regressor.joblib")
    print(f"\nSaved models to: {MODEL_DIR.resolve()}")
    return 0
NUM_COLS_EXPECTED = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "station_delay_mean_20",
    "station_delay_last_60min",   # ← NEW
]

if __name__ == "__main__":
    sys.exit(main())
