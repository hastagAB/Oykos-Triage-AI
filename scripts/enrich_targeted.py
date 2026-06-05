"""
Targeted catalog enrichment for specific symptom codes.
Loads existing enriched catalog (or base catalog), enriches only the
specified codes, and saves back. Run with:

    python scripts/enrich_targeted.py [CODE1 CODE2 ...]

If no codes given, defaults to the 4 symptoms responsible for GPT-5.5 misses.
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from symptom_extraction.config import get_api_key, load_config
from symptom_extraction.catalog.loader import load_catalog, ENRICHED_CATALOG, DEFAULT_CATALOG
from symptom_extraction.catalog.enrich import CatalogEnricher
from symptom_extraction.llm.openai_provider import OpenAIProvider

DEFAULT_TARGETS = ["SI040", "SI071", "SI051", "SI062"]


async def main(target_codes: list[str]):
    config = load_config()
    api_key = get_api_key(config.provider)

    # Always use OpenAI for enrichment (most capable for Italian medical text)
    from symptom_extraction.llm.openai_provider import OpenAIProvider
    llm = OpenAIProvider(api_key=api_key, default_model=config.frontier_model)

    # Load full catalog (enriched if it exists, base otherwise)
    catalog = load_catalog()
    targets = [s for s in catalog if s.code in target_codes]

    if not targets:
        print(f"No symptoms found for codes: {target_codes}")
        sys.exit(1)

    print(f"Enriching {len(targets)} symptoms:")
    for s in targets:
        print(f"  {s.code}: {s.label_it}")

    enricher = CatalogEnricher(llm, config)
    entries = await enricher.enrich_batch(targets, catalog)

    entry_map = {e.code: e for e in entries}

    # Load existing enriched catalog to merge into, or start from base
    if ENRICHED_CATALOG.exists():
        existing = json.loads(ENRICHED_CATALOG.read_text(encoding="utf-8"))
        existing_map = {s["code"]: s for s in existing}
    else:
        existing_map = {s.code: s.model_dump() for s in catalog}

    for code, entry in entry_map.items():
        if code in existing_map:
            existing_map[code]["examples_it"] = entry.examples_it
            existing_map[code]["synonyms_it"] = entry.synonyms_it
            existing_map[code]["negation_patterns_it"] = entry.negation_patterns_it
            existing_map[code]["disambiguation"] = entry.disambiguation
            existing_map[code]["safety_critical"] = entry.safety_critical
            print(f"  Updated {code}: {len(entry.examples_it)} examples, {len(entry.synonyms_it)} synonyms")
        else:
            print(f"  Warning: {code} not found in existing catalog")

    out = list(existing_map.values())
    ENRICHED_CATALOG.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved enriched catalog to {ENRICHED_CATALOG}")
    print(f"Total entries: {len(out)}")


if __name__ == "__main__":
    codes = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TARGETS
    asyncio.run(main(codes))
