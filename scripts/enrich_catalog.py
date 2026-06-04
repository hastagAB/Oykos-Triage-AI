"""Standalone script to generate the enriched symptom catalog."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from symptom_extraction.catalog.enrich import CatalogEnricher
from symptom_extraction.catalog.loader import load_catalog
from symptom_extraction.config import get_api_key, load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def main():
    config = load_config()

    if config.provider == "anthropic":
        from symptom_extraction.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=get_api_key("anthropic"),
            default_model=config.frontier_model,
        )
    else:
        from symptom_extraction.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            api_key=get_api_key("openai"),
            default_model=config.frontier_model,
        )

    catalog = load_catalog()
    enricher = CatalogEnricher(provider, config)
    enriched = await enricher.enrich_all(catalog)
    print(f"\nEnriched {len(enriched)} symptoms.")
    print(f"Symptoms with examples: {sum(1 for s in enriched if s.examples_it)}")
    print(f"Symptoms with synonyms: {sum(1 for s in enriched if s.synonyms_it)}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
