"""
Download the Telco Customer Churn dataset from Kaggle.

IMPORTANT
---------
Kaggle does NOT allow direct unauthenticated downloads via plain URLs.
You must have:
- A Kaggle account
- kaggle.json API token placed at ~/.kaggle/kaggle.json

Docs:
https://www.kaggle.com/docs/api

This script wraps the Kaggle CLI for reproducible dataset downloads.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DATASET = "blastchar/telco-customer-churn"
DEFAULT_OUT_DIR = Path("data/raw")


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    print(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Telco churn dataset from Kaggle.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory (relative to repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Downloading dataset from Kaggle")
    print(f"[info] Dataset: {DATASET}")
    print(f"[info] Output dir: {out_dir}")

    run([
        "kaggle",
        "datasets",
        "download",
        "-d", DATASET,
        "-p", str(out_dir),
        "--unzip"
    ])

    print("[ok] Dataset downloaded and extracted")
    print("[expected] CSV: Telco-Customer-Churn.csv")


if __name__ == "__main__":
    main()
