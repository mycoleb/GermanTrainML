from __future__ import annotations

import json
import sys
from pathlib import Path
import requests

DATASET = "piebro/deutsche-bahn-data"
OUT_DIR_DEFAULT = "data/raw"

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GermanTrainML/1.0"
}

def get_parquet_index(dataset: str) -> dict:
    # Hugging Face Hub API (returns parquet urls grouped by config/split)
    # Docs: https://huggingface.co/docs/dataset-viewer/en/parquet
    url = f"https://huggingface.co/api/datasets/{dataset}/parquet"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.json()

def choose_url(index: dict) -> tuple[str, str, str]:
    """
    Returns (config, split, url)
    Prefer: split 'train', otherwise first available split.
    """
    if not isinstance(index, dict) or not index:
        raise ValueError("Parquet index is empty or not a dict. Is the dataset public / available?")

    config = next(iter(index.keys()))
    splits = index[config]
    if not isinstance(splits, dict) or not splits:
        raise ValueError("No splits found in parquet index.")

    split = "train" if "train" in splits else next(iter(splits.keys()))
    urls = splits[split]
    if not urls:
        raise ValueError(f"No parquet URLs for config={config} split={split}")

    # download first shard only (smallest iteration loop)
    return config, split, urls[0]

def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=UA, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

def main(out_dir: str = OUT_DIR_DEFAULT) -> int:
    out_dir_p = Path(out_dir)
    index = get_parquet_index(DATASET)
    config, split, url = choose_url(index)

    print("HF parquet index loaded.")
    print(f"Chosen: dataset={DATASET} config={config} split={split}")
    print(f"URL: {url}")

    name = f"{DATASET.replace('/','__')}__{config}__{split}__shard0.parquet"
    out_path = out_dir_p / name

    print(f"Downloading -> {out_path}")
    download(url, out_path)
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
