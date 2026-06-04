"""Evaluation report generation — console and JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import AggregateMetrics, SymptomMetrics


def print_report(report) -> None:
    """Print a formatted evaluation report to the console."""
    agg: AggregateMetrics = report.aggregate
    per_symptom: dict[str, SymptomMetrics] = report.per_symptom

    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    print(f"\nMode:            {report.config.get('mode', '?')}")
    print(f"Provider:        {report.config.get('provider', '?')}")
    print(f"Model:           {report.config.get('frontier_model', '?')}")
    print(f"Total cases:     {agg.total_cases}")

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Macro F1':<25} {agg.macro_f1:>10.4f}")
    print(f"{'Micro F1':<25} {agg.micro_f1:>10.4f}")
    print(f"{'Macro Precision':<25} {agg.macro_precision:>10.4f}")
    print(f"{'Macro Recall':<25} {agg.macro_recall:>10.4f}")

    # Per-symptom table sorted by F1 ascending (worst first)
    sorted_symptoms = sorted(per_symptom.values(), key=lambda m: m.f1)

    print(f"\n{'='*70}")
    print("PER-SYMPTOM METRICS (sorted by F1, worst first)")
    print(f"{'='*70}")
    print(
        f"{'Code':<7} {'Label':<45} {'P':>5} {'R':>5} {'F1':>5} {'Supp':>5}"
    )
    print("-" * 75)
    for m in sorted_symptoms:
        if m.support == 0 and m.tp == 0 and m.fp == 0:
            continue
        label = m.label_it[:43] + ".." if len(m.label_it) > 45 else m.label_it
        print(
            f"{m.code:<7} {label:<45} {m.precision:>5.2f} {m.recall:>5.2f} "
            f"{m.f1:>5.2f} {m.support:>5}"
        )

    # Symptoms below recall floor
    if agg.symptoms_below_recall_floor:
        print(f"\n{'='*70}")
        print(
            f"SYMPTOMS BELOW RECALL FLOOR ({report.config.get('recall_floor', 0.85)})"
        )
        print(f"{'='*70}")
        for label in agg.symptoms_below_recall_floor:
            m = per_symptom.get(label)
            if m:
                print(f"  {m.code} {label}: recall={m.recall:.2f} (support={m.support})")
    else:
        print(f"\nAll symptoms meet the recall floor.")

    # Case type breakdown
    case_types: dict[str, list] = {}
    for cr in report.case_results:
        ct = cr.case_type or cr.section
        case_types.setdefault(ct, []).append(cr)

    if case_types:
        print(f"\n{'='*70}")
        print("BREAKDOWN BY CASE TYPE / SECTION")
        print(f"{'='*70}")
        print(f"{'Type/Section':<40} {'Cases':>6} {'Avg F1':>8}")
        print("-" * 56)
        for ct, cases in sorted(case_types.items()):
            avg_f1 = sum(c.metrics.f1 for c in cases) / len(cases) if cases else 0
            print(f"{ct[:38]:<40} {len(cases):>6} {avg_f1:>8.4f}")

    # Errors
    errors = [cr for cr in report.case_results if cr.error]
    if errors:
        print(f"\n{'='*70}")
        print(f"ERRORS ({len(errors)} cases)")
        print(f"{'='*70}")
        for cr in errors[:10]:
            print(f"  {cr.case_id}: {cr.error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Failed cases (FP or FN)
    failed = [cr for cr in report.case_results if cr.metrics.fp > 0 or cr.metrics.fn > 0]
    if failed:
        print(f"\n{'='*70}")
        print(f"SAMPLE FAILURES (showing first 15 of {len(failed)})")
        print(f"{'='*70}")
        for cr in failed[:15]:
            print(f"\n  [{cr.case_id}] {cr.message[:80]}...")
            if cr.metrics.fn > 0:
                missed = set(cr.expected) - set(cr.predicted)
                print(f"    MISSED: {', '.join(missed)}")
            if cr.metrics.fp > 0:
                false_pos = set(cr.predicted) - set(cr.expected)
                print(f"    FALSE+: {', '.join(false_pos)}")

    print(f"\n{'='*70}\n")


def save_json_report(report, output_path: str | Path) -> None:
    """Save the full evaluation report as JSON."""
    path = Path(output_path)

    per_symptom_data = {}
    for label, m in report.per_symptom.items():
        per_symptom_data[label] = m.model_dump()

    case_data = []
    for cr in report.case_results:
        case_data.append({
            "case_id": cr.case_id,
            "message": cr.message,
            "expected": cr.expected,
            "predicted": cr.predicted,
            "metrics": cr.metrics.model_dump(),
            "case_type": cr.case_type,
            "section": cr.section,
            "error": cr.error,
        })

    output = {
        "config": report.config,
        "aggregate": report.aggregate.model_dump(),
        "per_symptom": per_symptom_data,
        "cases": case_data,
    }

    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
