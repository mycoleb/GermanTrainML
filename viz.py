from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
import folium




STATIONS_PATH = Path("data/static/db_stations.csv")
def plot_worst_stations_map(
    df: pd.DataFrame,
    top_n: int = 200,
    out_html: str = "outputs/worst_stations_map.html"
):
    stations = load_station_lookup()

    # Require coordinates
    if "lat" not in stations.columns or "lon" not in stations.columns:
        raise ValueError(
            "Station lookup must include lat/lon columns. "
            "Update load_station_lookup() to return lat/lon."
        )

    if stations["lat"].isna().all() or stations["lon"].isna().all():
        raise ValueError(
            "Station lookup CSV doesn't appear to include usable lat/lon values.\n"
            "Ensure data/static/db_stations.csv has latitude/longitude columns."
        )

    # Compute worst stations
    worst = (
        df.groupby("station_id")["delay_minutes"]
          .mean()
          .reset_index()
          .rename(columns={"delay_minutes": "avg_delay"})
          .sort_values("avg_delay", ascending=False)
          .head(top_n)
    )

    # Join names + coords
    merged = worst.merge(stations, left_on="station_id", right_on="eva", how="left")
    merged = merged.dropna(subset=["lat", "lon"]).copy()

    if merged.empty:
        raise ValueError(
            "After joining station coords, no rows had lat/lon.\n"
            "Check that station_id values match the 'eva' codes in your CSV."
        )

    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce")
    merged = merged.dropna(subset=["lat", "lon"]).copy()

    # Center map
    center_lat = float(merged["lat"].median())
    center_lon = float(merged["lon"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Color scale (green -> red)
    min_d = float(merged["avg_delay"].min())
    max_d = float(merged["avg_delay"].max())
    span = max(max_d - min_d, 1e-9)

    def color_for(delay: float) -> str:
        t = (delay - min_d) / span  # 0..1
        r = int(255 * t)
        g = int(255 * (1 - t))
        return f"#{r:02x}{g:02x}00"

    # Plot points
    for _, r in merged.iterrows():
        delay = float(r["avg_delay"])
        name = r["name"] if pd.notna(r["name"]) else "Unknown"
        sid = r["station_id"]

        popup = f"{name} ({sid})<br>Avg delay: {delay:.2f} min"
        col = color_for(delay)

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(m)

    m.save(str(out_path))
    print(f"Wrote interactive map: {out_path}")

def load_station_lookup() -> pd.DataFrame:
    if not STATIONS_PATH.exists():
        raise FileNotFoundError(
            "Missing station lookup file.\n"
            "Place DB station metadata CSV at data/static/db_stations.csv"
        )

    df = pd.read_csv(STATIONS_PATH, dtype=str)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if "eva" not in df.columns or "name" not in df.columns:
        raise ValueError("Station file must contain 'eva' and 'name' columns")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    
    return df[["eva", "name", "lat", "lon"]].drop_duplicates()



DATA_PATH = Path("data/processed/dataset.parquet")
CLS_MODEL_PATH = Path("models/delay_classifier.joblib")
REG_MODEL_PATH = Path("models/delay_regressor.joblib")


# ----------------------------
# Data loading
# ----------------------------
def load_data(sample: int | None, seed: int = 42) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/processed/dataset.parquet. Run p.py first.")

    df = pd.read_parquet(DATA_PATH)

    if sample is not None and sample > 0 and sample < len(df):
        df = df.sample(sample, random_state=seed)

    return df


# ----------------------------
# Plot functions
# ----------------------------
def plot_delay_distribution(df: pd.DataFrame, max_minutes: int = 30):
    plt.figure(figsize=(10, 6))
    plt.hist(df["delay_minutes"], bins=100)
    plt.xlim(0, max_minutes)
    plt.title("Distribution of Train Delays (Germany)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Number of stops")
    plt.tight_layout()
    plt.show()


def plot_delay_by_hour(df: pd.DataFrame):
    hourly = df.groupby("hour")["delay_minutes"].mean()

    plt.figure(figsize=(10, 6))
    hourly.plot(marker="o")
    plt.title("Average Train Delay by Hour of Day")
    plt.xlabel("Hour of day")
    plt.ylabel("Average delay (minutes)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_delay_heatmap(df: pd.DataFrame):
    pivot = (
        df.assign(is_delayed=df["delay_minutes"] >= 5)
          .groupby(["dow", "hour"])["is_delayed"]
          .mean()
          .unstack()
    )

    plt.figure(figsize=(14, 6))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="P(delay ≥ 5 min)")
    plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xticks(range(24))
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    plt.title("Probability of Delay ≥ 5 Minutes (Heatmap)")
    plt.tight_layout()
    plt.show()


def plot_worst_stations(df: pd.DataFrame, top_n: int = 15):
    stations = load_station_lookup()

    worst = (
        df.groupby("station_id")["delay_minutes"]
          .mean()
          .reset_index()
          .sort_values("delay_minutes", ascending=False)
          .head(top_n)
    )

    worst = worst.merge(
        stations,
        left_on="station_id",
        right_on="eva",
        how="left"
    )

    worst["label"] = worst.apply(
        lambda r: f"{r['name']} ({r['station_id']})"
        if pd.notna(r["name"]) else r["station_id"],
        axis=1
    )

    plt.figure(figsize=(10, 6))
    plt.barh(worst["label"], worst["delay_minutes"])
    plt.xlabel("Average delay (minutes)")
    plt.title(f"Worst Stations by Average Delay (Top {top_n})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_pred_vs_actual(df: pd.DataFrame, max_minutes: int = 20):
    if not REG_MODEL_PATH.exists():
        raise FileNotFoundError("Missing models/delay_regressor.joblib. Run train.py first.")

    reg = load(REG_MODEL_PATH)
    df2 = df.dropna(subset=["delay_minutes"]).copy()

    X = df2.drop(columns=["delay_minutes", "is_delayed_5", "scheduled_time"])
    y_true = df2["delay_minutes"].astype(float)
    y_pred = reg.predict(X)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.2)
    plt.plot([0, max_minutes], [0, max_minutes])
    plt.xlim(0, max_minutes)
    plt.ylim(0, max_minutes)
    plt.xlabel("Actual delay (min)")
    plt.ylabel("Predicted delay (min)")
    plt.title("Predicted vs Actual Delay (Regressor)")
    plt.tight_layout()
    plt.show()


def plot_predicted_risk_by_hour(
    station_id: str,
    month: int,
    dow: int,
    is_weekend: int,
    station_delay_mean_20: float,
    category: str = "UNKNOWN",
    train_id: str = "UNKNOWN",
):
    if not CLS_MODEL_PATH.exists():
        raise FileNotFoundError("Missing models/delay_classifier.joblib. Run train.py first.")

    cls = load(CLS_MODEL_PATH)

    rows = []
    for hour in range(24):
        rows.append({
            "station_id": str(station_id),
            "category": str(category),
            "train_id": str(train_id),
            "hour": float(hour),
            "dow": float(dow),
            "month": float(month),
            "is_weekend": float(is_weekend),
            "station_delay_mean_20": float(station_delay_mean_20),
        })
    X = pd.DataFrame(rows)

    probs = cls.predict_proba(X)[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(range(24), probs, marker="o")
    plt.xlabel("Hour")
    plt.ylabel("P(delay ≥ 5 min)")
    plt.title("Predicted Delay Risk by Hour (Model Output)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--plot",
    required=True,
    choices=[
        "dist", "hour", "heatmap", "worst",
        "pred_vs_actual", "risk_by_hour",
        "map_worst"
    ],
    help="Which visualization to show"
)
    ap.add_argument(
    "--out",
    default="outputs/worst_stations_map.html",
    help="Output HTML file for map_worst"
)

    ap.add_argument("--sample", type=int, default=200000, help="Rows to sample for plots (0=all)")
    ap.add_argument("--seed", type=int, default=42)

    # risk_by_hour parameters
    ap.add_argument("--station-id", default="0370014")
    ap.add_argument("--category", default="UNKNOWN")
    ap.add_argument("--train-id", default="UNKNOWN")
    ap.add_argument("--month", type=int, default=7)
    ap.add_argument("--dow", type=int, default=0)
    ap.add_argument("--is-weekend", type=int, default=0)
    ap.add_argument("--station-delay-mean-20", type=float, default=3.0)

    # misc
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--max-minutes", type=int, default=30)

    args = ap.parse_args()

    sample = None if args.sample == 0 else args.sample
    df = load_data(sample=sample, seed=args.seed)

    if args.plot == "dist":
        plot_delay_distribution(df, max_minutes=args.max_minutes)
    elif args.plot == "hour":
        plot_delay_by_hour(df)
    elif args.plot == "heatmap":
        plot_delay_heatmap(df)
    elif args.plot == "worst":
        plot_worst_stations(df, top_n=args.top_n)
    elif args.plot == "pred_vs_actual":
        plot_pred_vs_actual(df, max_minutes=min(args.max_minutes, 60))
    elif args.plot == "risk_by_hour":
        plot_predicted_risk_by_hour(
            station_id=args.station_id,
            category=args.category,
            train_id=args.train_id,
            month=args.month,
            dow=args.dow,
            is_weekend=args.is_weekend,
            station_delay_mean_20=args.station_delay_mean_20,
        )
    
    elif args.plot == "map_worst":
        plot_worst_stations_map(df, top_n=args.top_n, out_html=args.out)

    else:
        raise RuntimeError("Unknown plot option")

if __name__ == "__main__":
    main()
