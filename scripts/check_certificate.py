#!/usr/bin/env python3
"""Check one ordered-cover certificate against its embedded instance."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fixedscan.checker import CertificateError, load_and_check, load_instance_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--instance", type=Path, help="optional independent instance file")
    args = parser.parse_args()
    try:
        instance = load_instance_file(args.instance) if args.instance else None
    except (CertificateError, OSError, ValueError, TypeError) as exc:
        result = {"accepted": False, "status": "rejected", "checked_nodes": 0, "cpu_seconds": 0.0, "error": str(exc)}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    result = load_and_check(args.certificate, instance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
