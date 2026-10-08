#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fixedscan.generate import write_collections

if __name__ == "__main__":
    summary = write_collections(ROOT / "cases")
    print(summary)
