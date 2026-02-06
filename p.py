from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def find_first_datafile() -> Path:
    # Prefer parquet (what you downloaded), fallback to csv
    pq = sorted(RAW_DIR.glob("*.parquet"))
    if pq:
        return pq[0]
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError("No .parquet or .csv found in data/raw.")

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def pick_col(cols: list[str], candidates: list[str]) -> str | None:
    """
    Return the first column whose normalized name matches any candidate normalized name,
    or contains a candidate token.
    """
    ncols = {norm(c): c for c in cols}
    # exact match first
    for cand in candidates:
        if cand in ncols:
            return ncols[cand]
    # containment fallback
    for c in cols:
        nc = norm(c)
        for cand in candidates:
            if cand in nc:
                return c
    return None

def to_utc_datetime(series: pd.Series) -> pd.Series:
    # robust datetime parse
    return pd.to_datetime(series, errors="coerce", utc=True)

def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

# -----------------------
# Main dataset builder
# -----------------------
def main() -> int:
    path = find_first_datafile()
    print(f"Reading: {path}")
    df = read_any(path)

    if df.empty:
        raise ValueError("Input file loaded but contains 0 rows.")
    # Sort only after we know what the time column is
    time_col = None
    for c in ["scheduled_time", "planned_time", "sched_time", "time", "timestamp", "datetime"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        # Print columns to help debugging
        print("Available columns:", list(df.columns))
        raise KeyError("No time column found in raw data (expected something like scheduled_time/planned_time/time).")

    df = df.sort_values(time_col)


#     df["station_delay_last_60min"] = (
#     df.groupby("station_id")
#       .rolling("60min", on="scheduled_time")["delay_minutes"]
#       .mean()
#       .reset_index(level=0, drop=True)
#       .astype("float32")
# )

    cols = list(df.columns)

    # Try to detect key columns from common conventions in GTFS-RT / DB / community datasets
    station_col = pick_col(cols, ["station_id", "eva", "eva_nr", "eva_id", "stop_id", "station", "bahnhof"])
    delay_col = pick_col(cols, ["delay_minutes", "delay_min", "delay", "lateness", "verspaetung", "verspaetung_min"])
    sched_time_col = pick_col(cols, ["scheduled_time", "planned_time", "schedule_time", "departure_scheduled",
                                    "arrival_scheduled", "pt", "planned_departure", "planned_arrival"])
    event_time_col = pick_col(cols, ["event_time", "actual_time", "realtime_time", "rt_time",
                                    "departure_actual", "arrival_actual", "actual_departure", "actual_arrival"])

    # Optional identifiers
    train_id_col = pick_col(cols, ["train_id", "trip_id", "journey_id", "zug_id", "rid", "run_id"])
    category_col = pick_col(cols, ["category", "product", "line", "train_category", "zugart"])

    # If delay_minutes isn't present, try to derive from actual - scheduled
    if delay_col is None and (sched_time_col is not None and event_time_col is not None):
        print("No delay column found; will derive delay_minutes from event_time - scheduled_time.")
        df["_scheduled_time"] = to_utc_datetime(df[sched_time_col])
        df["_event_time"] = to_utc_datetime(df[event_time_col])
        df["delay_minutes"] = (df["_event_time"] - df["_scheduled_time"]).dt.total_seconds() / 60.0
        delay_col = "delay_minutes"
        sched_time_col = "_scheduled_time"
    elif delay_col is not None:
        # parse scheduled time if present
        if sched_time_col is not None:
            df["_scheduled_time"] = to_utc_datetime(df[sched_time_col])
            sched_time_col = "_scheduled_time"
        else:
            # If we don't have scheduled time at all, we can still train, but we lose time features.
            # We'll create a placeholder NaT and warn.
            print("Warning: no scheduled_time column detected. Time-based features will be missing.")
            df["_scheduled_time"] = pd.NaT
            sched_time_col = "_scheduled_time"

        df["delay_minutes"] = safe_to_numeric(df[delay_col])
        delay_col = "delay_minutes"
    else:
        # Can't proceed
        raise ValueError(
            "Could not detect a delay column and could not derive delay from times.\n"
            f"Detected: station={station_col}, scheduled={sched_time_col}, event={event_time_col}, delay={delay_col}\n"
            f"Columns: {cols[:50]}{' ...' if len(cols) > 50 else ''}"
        )

    if station_col is None:
        print("Warning: no station column detected. Using a single dummy station_id.")
        df["station_id"] = "UNKNOWN"
        station_col = "station_id"
    else:
        df["station_id"] = df[station_col].astype("string")

    # Scheduled time column name already normalized to _scheduled_time
    df["scheduled_time"] = df[sched_time_col]

    # Clean
    df = df.dropna(subset=["station_id", "delay_minutes", "scheduled_time"]).copy()
    
    #  Ensure a unique index to prevent reindexing errors
    df = df.reset_index(drop=True)
    
    print(f"Rows with valid scheduled_time: {df['scheduled_time'].notna().sum()}")
    # If scheduled_time is present, we can add time features
    if pd.api.types.is_datetime64_any_dtype(df["scheduled_time"]):
        dt = df["scheduled_time"].dt
        df["hour"] = dt.hour.astype("Int16")
        df["dow"] = dt.dayofweek.astype("Int16")      # 0=Mon
        df["month"] = dt.month.astype("Int16")
        df["is_weekend"] = (df["dow"] >= 5).astype("Int8")
    else:
        # no time features
        df["hour"] = pd.Series([pd.NA] * len(df), dtype="Int16")
        df["dow"] = pd.Series([pd.NA] * len(df), dtype="Int16")
        df["month"] = pd.Series([pd.NA] * len(df), dtype="Int16")
        df["is_weekend"] = pd.Series([pd.NA] * len(df), dtype="Int8")

    # Rolling mean delay per station (sorted by scheduled_time when available)
    # Rolling mean delay per station (sorted by scheduled_time when available)
    sort_cols = ["station_id"]
    if pd.api.types.is_datetime64_any_dtype(df["scheduled_time"]):
        sort_cols.append("scheduled_time")

        # Sort AND reset index so assignments never try to reindex on duplicate labels
        df = df.sort_values(sort_cols).reset_index(drop=True)

        s20 = (
            df.groupby("station_id")["delay_minutes"]
            .rolling(window=20, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["station_delay_mean_20"] = pd.to_numeric(s20, errors="coerce").to_numpy(dtype="float32")

        s60 = (
            df.groupby("station_id")
            .rolling("60min", on="scheduled_time")["delay_minutes"]
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["station_delay_last_60min"] = pd.to_numeric(s60, errors="coerce").to_numpy(dtype="float32")
    else:
        df["station_delay_mean_20"] = np.nan
        df["station_delay_last_60min"] = np.nan


    # Targets
    df["is_delayed_5"] = (df["delay_minutes"] >= 5).astype("int8")

    # Optional categorical fields
    keep = [
    "station_id",
    "scheduled_time",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "station_delay_mean_20",
    "station_delay_last_60min",  # ← ADD THIS
    "delay_minutes",
    "is_delayed_5",
]


    if train_id_col is not None:
        df["train_id"] = df[train_id_col].astype("string")
        keep.insert(1, "train_id")

    if category_col is not None:
        df["category"] = df[category_col].astype("string")
        keep.insert(1, "category")

    out = df[keep].copy()
    out_path = OUT_DIR / "dataset.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Rows: {len(out):,}")
    print("Columns:", list(out.columns))
    print("Sample:")
    print(out.head(3).to_string(index=False))

    return 0

if __name__ == "__main__":
    sys.exit(main())
