from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

DATA_PATH = Path("data/processed/dataset.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/processed/dataset.parquet. Run p.py first.")

    print(f"Loading: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # Targets
    y_cls = df["is_delayed_5"].astype(int)
    y_reg = df["delay_minutes"].astype(float)

    # Features (drop raw targets + timestamp)
    drop_cols = {"is_delayed_5", "delay_minutes", "scheduled_time"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # Identify categorical vs numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "string" or X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print("Features:")
    print("  categorical:", cat_cols)
    print("  numeric    :", num_cols)

    # Numeric pipeline: replace NA -> -1, then cast to float32
    # Numeric pipeline: replace NA -> -1, then cast to float32
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("to_float", FunctionTransformer(lambda x: x.astype("float32"),
                                        feature_names_out="one-to-one")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )



    # Keep training fast; you can raise max_iter later
    cls = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=200)
    reg = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=200)

    cls_pipe = Pipeline([("pre", pre), ("model", cls)])
    reg_pipe = Pipeline([("pre", pre), ("model", reg)])

    # Train/test split
    # Stratify classification target so delayed/not-delayed ratio stays similar
    X_train, X_test, ycls_train, ycls_test, yreg_train, yreg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    print("Training classifier...")
    cls_pipe.fit(X_train, ycls_train)

    print("Training regressor...")
    reg_pipe.fit(X_train, yreg_train)

    # Evaluate
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

if __name__ == "__main__":
    sys.exit(main())
