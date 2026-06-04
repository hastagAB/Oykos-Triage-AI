"""Append hand-authored synthetic cases (synthetic_cases.py) to the master
test dataset. Idempotent: skips cases whose message already exists.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from synthetic_cases import all_records

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "test"
CATALOG_PATH = OUT_DIR / "symptom_catalog.json"
DATASET_PATH = OUT_DIR / "test_dataset.jsonl"
CSV_PATH = OUT_DIR / "test_dataset.csv"
STATS_PATH = OUT_DIR / "test_dataset_stats.md"


def load_existing() -> list[dict]:
    if not DATASET_PATH.exists():
        return []
    out = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    canon_labels = {s["label_it"] for s in catalog}

    existing = load_existing()
    seen_msgs = {r["message"].strip() for r in existing}
    seen_ids = {r["id"] for r in existing}
    print(f"Existing test cases: {len(existing)}")

    counters: Counter[str] = Counter()
    new_records: list[dict] = []
    for ctype, msg, gold, extras in all_records():
        msg = msg.strip()
        if not msg or msg in seen_msgs:
            continue
        seen_msgs.add(msg)
        for g in gold:
            if g not in canon_labels:
                raise SystemExit(f"Gold label not in catalog: {g!r} in case: {msg!r}")
        counters[ctype] += 1
        rid = f"syn_{ctype}_{counters[ctype]:04d}"
        while rid in seen_ids:
            counters[ctype] += 1
            rid = f"syn_{ctype}_{counters[ctype]:04d}"
        seen_ids.add(rid)
        rec = {
            "id": rid,
            "source": "synthetic:author",
            "section": f"synthetic_{ctype}",
            "message": msg,
            "expected_symptoms_canonical": gold,
            "expected_symptoms_unresolved": [],
            "expected_symptoms_raw": "; ".join(gold) if gold else "(none)",
            "all_labels_in_catalog": True,
            "case_type": ctype,
            "verdict_original_run": "",
            "verdict_retrained_run": "",
            "notes": "",
            **extras,
        }
        new_records.append(rec)

    print(f"New cases to append: {len(new_records)}")
    print(f"  by type: {dict(counters)}")

    with DATASET_PATH.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_records_full = load_existing()
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "source", "section", "case_type", "message",
            "expected_symptoms", "expected_symptoms_raw",
            "all_labels_in_catalog",
            "verdict_original_run", "verdict_retrained_run", "notes",
        ])
        for r in all_records_full:
            w.writerow([
                r["id"], r["source"], r["section"], r.get("case_type", ""),
                r["message"],
                "; ".join(r.get("expected_symptoms_canonical", [])),
                r.get("expected_symptoms_raw", ""),
                "yes" if r.get("all_labels_in_catalog") else "no",
                r.get("verdict_original_run", ""),
                r.get("verdict_retrained_run", ""),
                r.get("notes", ""),
            ])

    by_section = Counter(r["section"] for r in all_records_full)
    by_type = Counter(r.get("case_type", "(unspecified)") for r in all_records_full)
    cov: Counter[str] = Counter()
    for r in all_records_full:
        for lbl in r.get("expected_symptoms_canonical", []):
            cov[lbl] += 1
    not_tested = canon_labels - set(cov)

    lines = []
    lines.append("# Master Test Dataset - Coverage Report\n")
    lines.append(f"- canonical symptoms in catalog: **{len(catalog)}**")
    lines.append(f"- total test cases: **{len(all_records_full)}**\n")
    lines.append("## By case type\n")
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines.append("\n## By section\n")
    for k, v in sorted(by_section.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines.append("\n## Per-symptom support (positive gold counts)\n")
    for label, n in sorted(cov.items(), key=lambda x: -x[1]):
        lines.append(f"- {label}: {n}")
    lines.append(f"\n## Catalog symptoms with **zero** positive coverage ({len(not_tested)})\n")
    for s in sorted(not_tested):
        lines.append(f"- {s}")
    STATS_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDataset now: {len(all_records_full)} cases")
    print(f"  by type: {dict(by_type)}")
    print(f"  zero-coverage symptoms: {len(not_tested)}")


if __name__ == "__main__":
    main()
