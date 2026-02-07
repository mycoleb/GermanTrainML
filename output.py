from __future__ import annotations

from pathlib import Path
import matplotlib

# Headless backend so it saves PNGs reliably (no GUI needed)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Import your plot functions + data loader from viz.py (recommended)
# If your functions are inside viz.py, use:
from viz import (
    load_data,
    plot_delay_distribution,
    plot_delay_by_hour,
    plot_delay_heatmap,
    plot_worst_stations,
    plot_pred_vs_actual,
    plot_worst_stations_map,
    # plot_predicted_risk_map,  # enable once your station mapping is fixed
)

OUTDIR = Path("outputs")


def save_current_png(filename: str):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / filename
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()  # close so the next plot doesn't overwrite it
    print(f"Wrote PNG: {path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Load once
    df = load_data(sample=200_000, seed=42)

    # -----------------------
    # PNGs (static)
    # -----------------------
    plot_delay_distribution(df, max_minutes=30)
    save_current_png("delay_distribution.png")

    plot_delay_by_hour(df)
    save_current_png("delay_by_hour.png")

    plot_delay_heatmap(df)
    save_current_png("delay_heatmap.png")

    plot_worst_stations(df, top_n=20)
    save_current_png("worst_stations.png")

    plot_pred_vs_actual(df, max_minutes=30)
    save_current_png("pred_vs_actual.png")

    # -----------------------
    # HTML (interactive)
    # -----------------------
    plot_worst_stations_map(df, top_n=200, out_html=str(OUTDIR / "worst_map.html"))

    # Enable once your station join works for many stations:
    # plot_predicted_risk_map(
    #     df,
    #     out_html=str(OUTDIR / "pred_risk_map.html"),
    #     month=7, dow=0, is_weekend=0, hour=8, top_n=200
    # )

    print("\nDone. Open outputs/ to view PNGs + HTML maps.")


if __name__ == "__main__":
    main()
