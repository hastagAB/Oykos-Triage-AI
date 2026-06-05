"""Load the symptom catalog and produce EnrichedSymptom objects."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import EnrichedSymptom

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG = ROOT / "data" / "catalog" / "symptom_catalog.json"
ENRICHED_CATALOG = ROOT / "symptom_extraction" / "catalog" / "enriched_catalog.json"


def load_catalog(
    path: Path | str | None = None,
    enriched_path: Path | str | None = None,
) -> list[EnrichedSymptom]:
    """Load symptom catalog, preferring the enriched version if available.

    Falls back to the base catalog with empty enrichment fields.
    """
    ep = Path(enriched_path) if enriched_path else ENRICHED_CATALOG
    if ep.exists():
        raw = json.loads(ep.read_text(encoding="utf-8"))
        return [EnrichedSymptom.model_validate(s) for s in raw if s.get("label_it")]

    p = Path(path) if path else DEFAULT_CATALOG
    raw = json.loads(p.read_text(encoding="utf-8"))
    symptoms = []
    for s in raw:
        if not s.get("label_it"):
            continue
        symptoms.append(
            EnrichedSymptom(
                code=s["code"],
                label_it=s["label_it"],
                label_en=s.get("label_en", ""),
                triage_depth=s.get("triage_depth", ""),
                short_definition=s.get("short_definition", ""),
            )
        )
    return symptoms


def build_label_maps(
    catalog: list[EnrichedSymptom],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional code<->label_it maps.

    Returns:
        (code_to_label, label_to_code) dicts.
    """
    code_to_label = {s.code: s.label_it for s in catalog}
    label_to_code = {s.label_it: s.code for s in catalog}
    label_to_code.update({s.label_it.lower(): s.code for s in catalog})
    return code_to_label, label_to_code
