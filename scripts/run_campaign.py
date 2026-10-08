#!/usr/bin/env python3
"""Run the frozen and stress campaigns with one isolated worker at a time."""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fixedscan.checker import load_and_check
from fixedscan.model import Instance

METHODS = ("interval", "fresh", "syntactic", "residue", "exact-dp")
FIELDS = (
    "case", "family", "horizon", "method", "status",
    "production_cpu_seconds", "production_wall_seconds", "process_cpu_seconds",
    "nodes", "certificate_bytes", "checker_cpu_seconds", "checked_nodes",
    "peak_rss_kib", "queries", "encoding_bytes",
)


def run_task(case_path: Path, method: str, split: str, cert_path: Path) -> dict:
    internal_timeout = 20.0 if split == "frozen" else (5.0 if method != "exact-dp" else 12.0)
    hard_timeout = internal_timeout + 5.0
    max_nodes = 5000 if split == "frozen" else 20000
    cmd = [
        sys.executable, "-B", str(ROOT / "scripts" / "run_one.py"), str(case_path), method,
        "--max-nodes", str(max_nodes), "--timeout", str(internal_timeout),
    ]
    if method != "exact-dp":
        cmd += ["--certificate", str(cert_path)]
    env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUTF8"] = "1"; env["PYTHONDONTWRITEBYTECODE"] = "1"
    wall_start = time.monotonic()
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=hard_timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "kind": "timeout", "status": "unknown", "wall": time.monotonic() - wall_start,
            "error": "hard subprocess timeout",
            "worker_stdout": (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "worker_stderr": (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
        }
    wall = time.monotonic() - wall_start
    if cp.returncode != 0:
        return {"kind": "error", "status": "unknown", "wall": wall, "error": cp.stderr[-4000:],
                "worker_stdout": cp.stdout, "worker_stderr": cp.stderr}
    try:
        out = json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"kind": "error", "status": "unknown", "wall": wall, "error": f"bad task output: {exc}: {cp.stdout[-1000:]}",
                "worker_stdout": cp.stdout, "worker_stderr": cp.stderr}
    out["process_wall_seconds"] = wall
    out["worker_stdout"] = cp.stdout
    out["worker_stderr"] = cp.stderr
    return out


def row_from_raw(raw: dict) -> dict:
    task = raw["task"]; checker = raw.get("checker")
    status = task.get("status", "unknown")
    production_cpu = production_wall = process_cpu = None
    nodes = cert_bytes = queries = None
    if task.get("kind") == "cover":
        status = task["status"]; d = task["diagnostics"]
        production_cpu = d["cpu_seconds"]; production_wall = d["wall_seconds"]; process_cpu = d["cpu_seconds"]
        nodes = d["nodes"]; cert_bytes = task.get("certificate_bytes")
    elif task.get("kind") == "baseline":
        r = task["result"]; status = r["status"]
        production_cpu = r["cpu_seconds"]; production_wall = r["wall_seconds"]; process_cpu = r["cpu_seconds"]; queries = r["queries"]
    return {
        "case": raw["case"], "family": raw["family"], "horizon": raw["horizon"], "method": raw["method"], "status": status,
        "production_cpu_seconds": production_cpu, "production_wall_seconds": production_wall, "process_cpu_seconds": process_cpu,
        "nodes": nodes, "certificate_bytes": cert_bytes,
        "checker_cpu_seconds": checker["cpu_seconds"] if checker else None,
        "checked_nodes": checker["checked_nodes"] if checker else None,
        "peak_rss_kib": task.get("peak_rss_kib"), "queries": queries, "encoding_bytes": None,
    }


def collect_split(split: str, results_root: Path | None = None) -> list[dict]:
    results_root = results_root if results_root is not None else ROOT / "results"
    rows = []
    for path in sorted((results_root / "raw" / split).glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(row_from_raw(json.load(f)))
    rows.sort(key=lambda r: (r["case"], METHODS.index(r["method"])))
    metrics_path = results_root / f"{split}-metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    summarize(split, rows, results_root)
    return rows


def run_split(split: str, start: int = 0, count: int | None = None, results_root: Path | None = None) -> list[dict]:
    results_root = results_root if results_root is not None else ROOT / "results"
    case_dir = ROOT / "cases" / split
    raw_dir = results_root / "raw" / split
    cert_dir = results_root / "certificates" / split
    raw_dir.mkdir(parents=True, exist_ok=True); cert_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    all_cases = sorted(case_dir.glob("*.json"))
    cases = all_cases[start:] if count is None else all_cases[start:start + count]
    for case_no, case_path in enumerate(cases, start + 1):
        inst = Instance.load(case_path)
        for method in METHODS:
            raw_path = raw_dir / f"{inst.name}-{method}.json"
            if raw_path.exists():
                with raw_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                rows.append(row_from_raw(existing))
                continue
            cert_path = cert_dir / f"{inst.name}-{method}.json"
            if cert_path.exists():
                raise FileExistsError(f"orphan certificate must not be overwritten: {cert_path}")
            task = run_task(case_path, method, split, cert_path)
            if task.get("certificate"):
                # Historical records use repository-relative results/... paths;
                # fresh external campaigns use paths relative to their output.
                base = ROOT if results_root == ROOT / "results" else results_root
                task["certificate"] = cert_path.relative_to(base).as_posix()
            checker = None
            status = task.get("status", "unknown")
            production_cpu = production_wall = process_cpu = None
            nodes = cert_bytes = queries = None
            if task.get("kind") == "cover":
                status = task["status"]
                d = task["diagnostics"]
                production_cpu = d["cpu_seconds"]; production_wall = d["wall_seconds"]
                process_cpu = d["cpu_seconds"]
                nodes = d["nodes"]; cert_bytes = task.get("certificate_bytes")
                if status != "unknown":
                    checker = load_and_check(cert_path, inst)
                    if not checker["accepted"]:
                        raise RuntimeError(f"checker rejected {cert_path}: {checker}")
            elif task.get("kind") == "baseline":
                r = task["result"]; status = r["status"]
                production_cpu = r["cpu_seconds"]; production_wall = r["wall_seconds"]
                process_cpu = r["cpu_seconds"]; queries = r["queries"]
            else:
                status = "unknown"
            if split == "frozen":
                expected = "safe" if bool(inst.metadata["oracle_safe"]) else "violated"
                if status != expected:
                    raise RuntimeError(f"frozen disagreement for {inst.name}/{method}: {status} != {expected}; task={task}")
            raw = {
                "case": inst.name, "family": inst.family, "horizon": inst.horizon,
                "method": method, "task": task, "checker": checker,
            }
            with raw_path.open("w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2, sort_keys=True); f.write("\n")
            row = {
                "case": inst.name, "family": inst.family, "horizon": inst.horizon,
                "method": method, "status": status,
                "production_cpu_seconds": production_cpu,
                "production_wall_seconds": production_wall,
                "process_cpu_seconds": process_cpu,
                "nodes": nodes,
                "certificate_bytes": cert_bytes,
                "checker_cpu_seconds": checker["cpu_seconds"] if checker else None,
                "checked_nodes": checker["checked_nodes"] if checker else None,
                "peak_rss_kib": task.get("peak_rss_kib"),
                "queries": queries,
                "encoding_bytes": None,
            }
            rows.append(row)
        print(f"[{split}] {case_no}/{len(all_cases)} {inst.name}", flush=True)
    return rows


def summarize(split: str, rows: list[dict], results_root: Path | None = None) -> None:
    results_root = results_root if results_root is not None else ROOT / "results"
    summary: dict[str, object] = {"split": split, "instances": len({r['case'] for r in rows}), "methods": {}}
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
            "status_counts": {s: sum(r["status"] == s for r in rs) for s in ("safe", "violated", "unknown")},
            "total_nodes": sum(int(r["nodes"] or 0) for r in rs),
            "total_certificate_bytes": sum(int(r["certificate_bytes"] or 0) for r in rs),
            "producer_cpu_seconds": sum(float(r["production_cpu_seconds"] or 0) for r in rs),
            "checker_cpu_seconds": sum(float(r["checker_cpu_seconds"] or 0) for r in rs),
        }
    with (results_root / f"{split}-summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True); f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("frozen", "stress", "all"), default="all")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int)
    ap.add_argument("--out", type=Path, help="fresh campaign output; preserves shipped measurements")
    args = ap.parse_args()
    results_root = ROOT / "results"
    if args.out is not None:
        results_root = args.out.resolve()
        if results_root == (ROOT / "results").resolve() or (ROOT / "results").resolve() in results_root.parents:
            ap.error("--out must not be inside the shipped results tree")
        results_root.mkdir(parents=True, exist_ok=False)
    splits = ("frozen", "stress") if args.split == "all" else (args.split,)
    for split in splits:
        run_split(split, args.start, args.count, results_root); collect_split(split, results_root)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
