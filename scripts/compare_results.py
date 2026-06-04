"""Compare evaluation results across models."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

files = list((ROOT / "data" / "test").glob("eval_*.json"))
if not files:
    print("No evaluation files found.")
    sys.exit(1)

print(f"{'Model':<30} {'Micro F1':>10} {'Precision':>10} {'Recall':>10} {'Below Floor':>12}")
print("-" * 75)

for f in sorted(files):
    data = json.loads(f.read_text(encoding="utf-8"))
    model = data.get("config", {}).get("frontier_model", f.stem)
    a = data["aggregate"]
    below = len(a.get("symptoms_below_recall_floor", []))
    print(f"{model:<30} {a['micro_f1']:>10.4f} {a['macro_precision']:>10.4f} {a['macro_recall']:>10.4f} {below:>12}")
