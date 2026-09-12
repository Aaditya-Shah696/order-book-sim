"""Fetch the free LOBSTER AAPL 2012-06-21 top-of-book sample from its Hugging Face mirror.

Usage: uv run python scripts/fetch_lobster_sample.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/"
    "LOBSTER_SampleFile_AAPL_2012-06-21_1"
)
FILES = {
    "message.csv": f"{BASE_URL}/AAPL_2012-06-21_34200000_57600000_message_1.csv",
    "orderbook.csv": f"{BASE_URL}/AAPL_2012-06-21_34200000_57600000_orderbook_1.csv",
}
OUT_DIR = Path(__file__).parent.parent / "data" / "lobster"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        out_path = OUT_DIR / filename
        print(f"fetching {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)
        size_mb = out_path.stat().st_size / 1_000_000
        print(f"  {size_mb:.1f} MB")
    print(f"done, see {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        sys.exit(1)
