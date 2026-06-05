"""
Model comparison — simple accuracy: did the model get the message right or not?
A message is CORRECT if the model found every symptom the parent mentioned
and didn't add any that weren't there. Otherwise it's WRONG.
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"

MODEL_LABELS = {
    # OpenAI
    "gpt-5.5-2026-04-23":       "GPT-5.5            (OpenAI, flagship)",
    "gpt-5.4":                  "GPT-5.4             (OpenAI, fast)",
    "gpt-5.4-mini":             "GPT-5.4 Mini        (OpenAI, budget)",
    "gpt-5.4-nano":             "GPT-5.4 Nano        (OpenAI, cheapest)",
    # Anthropic — Sonnet line
    "claude-sonnet-4-6":        "Claude Sonnet 4.6   (Anthropic, latest)",
    "claude-sonnet-4-5":        "Claude Sonnet 4.5   (Anthropic)",
    "claude-sonnet-4-20250514": "Claude Sonnet 4     (Anthropic)",
    # Anthropic — Opus line
    "claude-opus-4-8":          "Claude Opus 4.8     (Anthropic, flagship)",
    "claude-opus-4-6":          "Claude Opus 4.6     (Anthropic)",
    "claude-opus-4-5":          "Claude Opus 4.5     (Anthropic)",
}

# Skip intermediate experiment runs (prompt regression, catalog-only test)
EXCLUDED_FILES = {"eval_v5_gpt55.json", "eval_v5_catalog_only.json"}

OPENAI_MODELS = {"gpt-5.5-2026-04-23", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}
ANTHROPIC_MODELS = {
    "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4-20250514",
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5",
}


def label(model_id):
    return MODEL_LABELS.get(model_id, model_id)


def bar(n, total, width=30):
    filled = round((n / total) * width) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def pct(n, total):
    return f"{n / total * 100:.1f}%" if total else "0.0%"


def analyse(data_file):
    path = EVAL_DIR / data_file
    data = json.loads(path.read_text(encoding="utf-8"))
    model_id = data.get("config", {}).get("frontier_model", path.stem)
    provider = data.get("config", {}).get("provider", "unknown")
    cases = data.get("cases", data.get("case_results", []))

    correct, missed_only, extra_only, both_errors = [], [], [], []
    for c in cases:
        m = c.get("metrics", {})
        fp, fn = m.get("fp", 0), m.get("fn", 0)
        if fp == 0 and fn == 0:
            correct.append(c)
        elif fn > 0 and fp == 0:
            missed_only.append(c)
        elif fp > 0 and fn == 0:
            extra_only.append(c)
        else:
            both_errors.append(c)

    return {
        "file": data_file,
        "model_id": model_id,
        "provider": provider,
        "label": label(model_id),
        "total": len(cases),
        "correct": len(correct),
        "wrong": len(cases) - len(correct),
        "missed_only": len(missed_only),   # got all extras right but missed a symptom
        "extra_only": len(extra_only),     # found everything but added something extra
        "both": len(both_errors),          # both missed and added wrong
        "correct_cases": correct,
        "wrong_cases": missed_only + extra_only + both_errors,
    }


def load_all():
    files = sorted(EVAL_DIR.glob("eval_*.json"))
    if not files:
        print("No evaluation files found in data/eval/")
        sys.exit(1)
    # Collect all valid results, then deduplicate keeping best run per model
    all_results = []
    for f in files:
        if f.name in EXCLUDED_FILES:
            continue
        try:
            all_results.append(analyse(f.name))
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    # Keep the highest-accuracy result per model
    best_per_model = {}
    for r in all_results:
        mid = r["model_id"]
        if mid not in best_per_model or r["correct"] > best_per_model[mid]["correct"]:
            best_per_model[mid] = r
    results = list(best_per_model.values())
    results.sort(key=lambda r: r["correct"], reverse=True)
    return results


def print_model_card(r, rank):
    rank_str = ["BEST", "2ND", "3RD", "4TH", "5TH", "6TH"][rank] if rank < 6 else f"{rank+1}TH"
    total = r["total"]

    print(f"  [{rank_str}]  {r['label']}")
    print()
    print(f"  Accuracy   {bar(r['correct'], total)}  {r['correct']} / {total} correct  ({pct(r['correct'], total)})")
    print()

    if r["wrong"] == 0:
        print("  No errors.")
    else:
        print(f"  Errors: {r['wrong']} messages wrong")
        if r["missed_only"]:
            print(f"    - {r['missed_only']} times: missed a symptom the parent mentioned")
        if r["extra_only"]:
            print(f"    - {r['extra_only']} times: found everything but added an extra symptom")
        if r["both"]:
            print(f"    - {r['both']} times: both missed one and added one wrong")
    print()

    # Show a few wrong cases
    if r["wrong_cases"]:
        print(f"  Example mistakes:")
        for c in r["wrong_cases"][:3]:
            msg = c["message"][:80].replace("\n", " ")
            expected = ", ".join(c["expected"]) or "(none)"
            predicted = ", ".join(c["predicted"]) or "(none)"
            print(f"    Message : {msg}...")
            print(f"    Expected: {expected}")
            print(f"    Got     : {predicted}")
            print()


def main():
    results = load_all()

    # Split into OpenAI and Anthropic
    openai_results = [r for r in results if r["model_id"] in OPENAI_MODELS]
    anthropic_results = [r for r in results if r["model_id"] in ANTHROPIC_MODELS]
    other_results = [r for r in results if r["model_id"] not in OPENAI_MODELS and r["model_id"] not in ANTHROPIC_MODELS]

    total = results[0]["total"] if results else 860

    print()
    print("=" * 70)
    print("  OYKOS TRIAGE AI — MODEL ACCURACY REPORT")
    print("=" * 70)
    print(f"  Test: {total} real parent messages (Italian)")
    print(f"  Grading: message is CORRECT only if every symptom is right")
    print(f"           — nothing missed, nothing added by mistake")
    print()

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if openai_results:
        print("=" * 70)
        print("  OPENAI MODELS")
        print("=" * 70)
        for i, r in enumerate(openai_results):
            print("-" * 70)
            print_model_card(r, i)

    # ── Anthropic ─────────────────────────────────────────────────────────────
    if anthropic_results:
        print("=" * 70)
        print("  ANTHROPIC MODELS")
        print("=" * 70)
        for i, r in enumerate(anthropic_results):
            print("-" * 70)
            print_model_card(r, i)

    if other_results:
        print("=" * 70)
        print("  OTHER MODELS")
        print("=" * 70)
        for i, r in enumerate(other_results):
            print("-" * 70)
            print_model_card(r, i)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Model':<38}  {'Correct':>10}  {'Wrong':>6}  {'Accuracy':>9}")
    print(f"  {'-'*38}  {'-'*10}  {'-'*6}  {'-'*9}")
    for r in results:
        print(
            f"  {r['label'][:38]:<38}"
            f"  {r['correct']:>4} / {r['total']:<4}"
            f"  {r['wrong']:>6}"
            f"  {pct(r['correct'], r['total']):>9}"
        )

    # ── Best of each provider ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    print()
    if openai_results:
        best_oa = openai_results[0]
        print(f"  Best OpenAI model   : {best_oa['label'].strip()}")
        print(f"                        {best_oa['correct']} / {best_oa['total']} correct ({pct(best_oa['correct'], best_oa['total'])})")
        print()
    if anthropic_results:
        best_an = anthropic_results[0]
        print(f"  Best Anthropic model: {best_an['label'].strip()}")
        print(f"                        {best_an['correct']} / {best_an['total']} correct ({pct(best_an['correct'], best_an['total'])})")
        print()

    if openai_results and anthropic_results:
        best_oa = openai_results[0]
        best_an = anthropic_results[0]
        diff = best_oa["correct"] - best_an["correct"]
        if diff > 0:
            print(f"  GPT-5.5 handles {diff} more messages correctly than Claude Sonnet 4.6.")
        elif diff < 0:
            print(f"  Claude Sonnet 4.6 handles {-diff} more messages correctly than GPT-5.5.")
        else:
            print(f"  Both top models are equally accurate.")
        print()
    print()


if __name__ == "__main__":
    main()
