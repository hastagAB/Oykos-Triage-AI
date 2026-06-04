"""Standalone script to pre-compute symptom embedding vectors."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from symptom_extraction.catalog.loader import load_catalog
from symptom_extraction.config import load_config
from symptom_extraction.embeddings.index import SymptomVectorIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    config = load_config()
    catalog = load_catalog()
    print(f"Building embeddings for {len(catalog)} symptoms...")

    index = SymptomVectorIndex(catalog, config)
    index.build_and_save()

    print(f"\nDone. Testing with sample query...")
    results = index.query("il bambino ha la febbre alta")
    for code, score in results[:5]:
        label = next((s.label_it for s in catalog if s.code == code), "?")
        print(f"  {code} {label}: {score:.4f}")


if __name__ == "__main__":
    main()
