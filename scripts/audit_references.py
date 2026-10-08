#!/usr/bin/env python3
"""Validate the shipped scholarly-reference provenance inventory."""
from __future__ import annotations
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference-audit.csv"
OUT = ROOT / "results" / "reference-audit.json"
REQUIRED = {
    "bib_key", "category", "title", "authors", "year", "venue", "doi",
    "canonical_record", "paper_cited", "manuscript_role", "metadata_check",
    "content_check", "access_date", "status", "notes",
}

def main() -> int:
    errors: list[str] = []
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    missing_columns = sorted(REQUIRED - fields)
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    if len(rows) < 30:
        errors.append(f"only {len(rows)} manuscript references; expected at least 30")
    keys = [r.get("bib_key", "") for r in rows]
    for key, count in Counter(keys).items():
        if not key or count != 1:
            errors.append(f"invalid/duplicate bib key {key!r}: {count}")
    identifiers: list[str] = []
    for idx, row in enumerate(rows, 2):
        for name in REQUIRED - {"doi"}:
            if not row.get(name, "").strip():
                errors.append(f"row {idx}: empty {name}")
        url = row.get("canonical_record", "")
        if not re.fullmatch(r"https://[^\s]+", url):
            errors.append(f"row {idx}: non-HTTPS canonical record")
        ident = (row.get("doi") or url).casefold().rstrip("/")
        identifiers.append(ident)
        if row.get("paper_cited") != "yes" or row.get("status") != "verified":
            errors.append(f"row {idx}: reference is not marked cited and verified")
        if re.search(r"example\.|placeholder|todo|tbd|dummy", " ".join(row.values()), re.I):
            errors.append(f"row {idx}: placeholder token")
    for ident, count in Counter(identifiers).items():
        if count != 1:
            errors.append(f"duplicate identifier {ident}: {count}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "manuscript_references": len(rows),
        "unique_identifiers": len(set(identifiers)),
        "all_cited": all(r.get("paper_cited") == "yes" for r in rows),
        "all_rows_recorded_as_verified": all(r.get("status") == "verified" for r in rows),
        "scope": "Inventory and recorded provenance fields only; no online or full-text verification is performed by this command.",
        "errors": errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
