from __future__ import annotations

from pathlib import Path
import pandas as pd

STATIONS_PATH = Path("data/static/db_stations.csv")


def _norm_eva7(x: str) -> str:
    # DB EVA is usually 7 digits (no leading 0)
    s = str(x).strip()
    s = s.lstrip("0")  # 08000207 -> 8000207
    return s.zfill(7)


def _norm_eva8(x: str) -> str:
    # your dataset sometimes stores it as 8 digits
    s = str(x).strip()
    return s.zfill(8)


def load_station_lookup(path: Path = STATIONS_PATH) -> pd.DataFrame:
    """
    Returns a lookup table with multiple normalized keys so joins succeed
    even if inputs are 7- or 8-digit variants.
    Requires columns: eva, name (lat/lon optional).
    """
    if not path.exists():
        raise FileNotFoundError(
            "Missing station lookup file.\n"
            "Place DB station metadata CSV at data/static/db_stations.csv"
        )

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.lower().strip() for c in df.columns]

    if "eva" not in df.columns or "name" not in df.columns:
        raise ValueError("Station file must contain columns: eva, name")

    df["eva7"] = df["eva"].map(_norm_eva7)
    df["eva8"] = df["eva"].map(_norm_eva8)

    # keep optional coords if present
    keep = ["eva", "eva7", "eva8", "name"]
    for c in ("lat", "lon"):
        if c in df.columns:
            keep.append(c)

    return df[keep].drop_duplicates()


def add_station_names(
    df: pd.DataFrame,
    station_id_col: str = "station_id",
    out_col: str = "station_name",
    path: Path = STATIONS_PATH,
) -> pd.DataFrame:
    """
    Adds df[out_col] by joining station IDs to station names.
    Works whether df has 7-digit or 8-digit station IDs.
    """
    stations = load_station_lookup(path)

    tmp = df.copy()
    tmp["_sid7"] = tmp[station_id_col].map(_norm_eva7)
    tmp["_sid8"] = tmp[station_id_col].map(_norm_eva8)

    # Try join on 7-digit normalized key
    out = tmp.merge(stations[["eva7", "name"]], left_on="_sid7", right_on="eva7", how="left")
    out.rename(columns={"name": out_col}, inplace=True)

    # Fill misses by joining on 8-digit normalized key
    miss = out[out_col].isna()
    if miss.any():
        out2 = tmp.loc[miss].merge(stations[["eva8", "name"]], left_on="_sid8", right_on="eva8", how="left")
        out.loc[miss, out_col] = out2["name"].values

    return out.drop(columns=["_sid7", "_sid8"], errors="ignore")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Lookup station name from EVA/station_id.")
    ap.add_argument("station_id", help="EVA / station_id (7 or 8 digits)")
    args = ap.parse_args()

    stations = load_station_lookup()
    sid7 = _norm_eva7(args.station_id)
    sid8 = _norm_eva8(args.station_id)

    row = stations[(stations["eva7"] == sid7) | (stations["eva8"] == sid8)].head(1)
    if row.empty:
        print("No match.")
    else:
        print(row.iloc[0]["name"])


if __name__ == "__main__":
    main()