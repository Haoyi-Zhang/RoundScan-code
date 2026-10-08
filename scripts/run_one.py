#!/usr/bin/env python3
"""Single isolated campaign task; invoked by run_campaign.py."""
from __future__ import annotations
import argparse
import json
try:
    import resource
except ImportError:  # The recorded Unix RSS metric is unavailable on Windows.
    resource = None
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fixedscan.baseline import exact_baseline
from fixedscan.model import Instance
from fixedscan.producer import produce, save_certificate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("method")
    ap.add_argument("--certificate")
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()
    inst = Instance.load(args.case)
    if args.method == "exact-dp":
        result = exact_baseline(inst, timeout_seconds=args.timeout)
        out = {
            "kind": "baseline",
            "result": result,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
        }
    else:
        cert = produce(inst, args.method, max_nodes=args.max_nodes, timeout_seconds=args.timeout)
        cert_bytes = None
        if cert["status"] != "unknown" and args.certificate:
            save_certificate(cert, args.certificate)
            cert_bytes = Path(args.certificate).stat().st_size
        out = {
            "kind": "cover",
            "certificate": args.certificate if cert_bytes is not None else None,
            "certificate_bytes": cert_bytes,
            "status": cert["status"],
            "diagnostics": cert["diagnostics"],
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
        }
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
