"""Metrics computation — per-case, per-symptom, and aggregate."""

from __future__ import annotations

from collections import defaultdict

from ..models import AggregateMetrics, CaseMetrics, EnrichedSymptom, SymptomMetrics


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def compute_case_metrics(
    predicted: list[str],
    expected: list[str],
    negated_symptom: str | None = None,
) -> CaseMetrics:
    """Compute case-level metrics.

    For negation cases: the negated_symptom must NOT appear in predictions.
    Other symptoms the model finds are not penalized since the message
    may legitimately contain them.
    """
    pred_set = set(predicted)
    exp_set = set(expected)

    if negated_symptom is not None:
        if negated_symptom in pred_set:
            return CaseMetrics(
                tp=0, fp=1, fn=0,
                precision=0.0, recall=1.0, f1=0.0,
                predicted=predicted, expected=expected,
            )
        return CaseMetrics(
            tp=0, fp=0, fn=0,
            precision=1.0, recall=1.0, f1=1.0,
            predicted=predicted, expected=expected,
        )

    if not exp_set and not pred_set:
        return CaseMetrics(
            tp=0, fp=0, fn=0,
            precision=1.0, recall=1.0, f1=1.0,
            predicted=predicted, expected=expected,
        )

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return CaseMetrics(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        predicted=predicted, expected=expected,
    )


def compute_symptom_metrics(
    all_predictions: list[list[str]],
    all_golds: list[list[str]],
    catalog: list[EnrichedSymptom],
) -> dict[str, SymptomMetrics]:
    """Compute per-symptom TP/FP/FN/precision/recall/F1 across all cases."""
    code_to_label = {s.code: s.label_it for s in catalog}

    tp_counts: dict[str, int] = defaultdict(int)
    fp_counts: dict[str, int] = defaultdict(int)
    fn_counts: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for preds, golds in zip(all_predictions, all_golds):
        pred_set = set(preds)
        gold_set = set(golds)

        for label in gold_set:
            support[label] += 1
            if label in pred_set:
                tp_counts[label] += 1
            else:
                fn_counts[label] += 1

        for label in pred_set:
            if label not in gold_set:
                fp_counts[label] += 1

    all_labels = set(tp_counts) | set(fp_counts) | set(fn_counts) | set(support)
    label_to_code = {s.label_it: s.code for s in catalog}

    result = {}
    for label in all_labels:
        code = label_to_code.get(label, "?")
        tp = tp_counts[label]
        fp = fp_counts[label]
        fn = fn_counts[label]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        result[label] = SymptomMetrics(
            code=code, label_it=label,
            tp=tp, fp=fp, fn=fn,
            precision=p, recall=r, f1=f1,
            support=support[label],
        )

    return result


def compute_aggregate_metrics(
    per_symptom: dict[str, SymptomMetrics],
    recall_floor: float = 0.85,
    min_support: int = 8,
    total_cases: int = 0,
) -> AggregateMetrics:
    """Macro-F1, micro-F1, and symptoms below recall floor."""
    total_tp = sum(m.tp for m in per_symptom.values())
    total_fp = sum(m.fp for m in per_symptom.values())
    total_fn = sum(m.fn for m in per_symptom.values())

    micro_p = _safe_div(total_tp, total_tp + total_fp)
    micro_r = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)

    symptoms_with_support = [
        m for m in per_symptom.values() if m.support > 0
    ]
    if symptoms_with_support:
        macro_p = sum(m.precision for m in symptoms_with_support) / len(symptoms_with_support)
        macro_r = sum(m.recall for m in symptoms_with_support) / len(symptoms_with_support)
        macro_f1 = sum(m.f1 for m in symptoms_with_support) / len(symptoms_with_support)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    below_floor = [
        m.label_it
        for m in per_symptom.values()
        if m.support >= min_support and m.recall < recall_floor
    ]

    return AggregateMetrics(
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        total_cases=total_cases,
        symptoms_below_recall_floor=sorted(below_floor),
    )
