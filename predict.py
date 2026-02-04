from utils import to_float32

# from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from joblib import load

MODEL_DIR = Path("models")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-id", required=True)
    ap.add_argument("--scheduled-time", required=True, help="ISO8601 like 2024-07-01T12:30:00Z")
    ap.add_argument("--category", default="")
    ap.add_argument("--train-id", default="")
    ap.add_argument("--station-delay-mean-20", type=float, default=0.0)
    args = ap.parse_args()

    cls = load(MODEL_DIR / "delay_classifier.joblib")
    reg = load(MODEL_DIR / "delay_regressor.joblib")

    t = pd.to_datetime(args.scheduled_time, utc=True)

    row = {
        "station_id": str(args.station_id),
        "category": str(args.category),
        "train_id": str(args.train_id),
        "hour": int(t.hour),
        "dow": int(t.dayofweek),
        "month": int(t.month),
        "is_weekend": int(t.dayofweek >= 5),
        "station_delay_mean_20": float(args.station_delay_mean_20),
    }
    X = pd.DataFrame([row])

    p_delay5 = float(cls.predict_proba(X)[:, 1][0])
    delay_min = float(reg.predict(X)[0])

    print("=== PREDICTION ===")
    print(f"P(delay ≥ 5 min): {p_delay5:.3f}")
    print(f"Predicted delay (min): {delay_min:.2f}")

if __name__ == "__main__":
    main()
