"""Bootstrap enriched catalog with LLM-generated synonyms, phrasings, and negation patterns."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider
from ..models import EnrichedSymptom, PipelineConfig

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent / "enriched_catalog.json"

ENRICHMENT_SYSTEM_PROMPT = """\
You are a medical linguist specializing in Italian pediatric healthcare communication.

Your task is to enrich a pediatric symptom catalog entry with realistic Italian \
parent phrasings, synonyms, negation patterns, and disambiguation notes.

## Context
Parents describe their children's symptoms in colloquial Italian via WhatsApp \
messages to a pediatrician. The language is informal, often incomplete, may \
include dialect, misspellings, and abbreviations.

## Output format
For each symptom, provide:
1. examples_it: 8-10 realistic Italian phrases a parent might write \
(WhatsApp style, colloquial, varied formulations)
2. synonyms_it: 5-8 Italian synonym surface forms (single words or short \
phrases that refer to this symptom)
3. negation_patterns_it: 4-6 Italian phrases that NEGATE this symptom \
(e.g., "non ha la febbre", "niente tosse")
4. disambiguation: A short note explaining how to distinguish this symptom \
from similar/adjacent symptoms in the catalog
5. safety_critical: true if this symptom could indicate a life-threatening \
condition requiring immediate attention

Respond using the provided JSON schema.
"""


class EnrichmentEntry(BaseModel):
    code: str
    examples_it: list[str] = Field(
        description="8-10 realistic Italian parent phrasings"
    )
    synonyms_it: list[str] = Field(
        description="5-8 Italian synonym surface forms"
    )
    negation_patterns_it: list[str] = Field(
        description="4-6 Italian negation phrases for this symptom"
    )
    disambiguation: str = Field(
        description="How to distinguish from similar symptoms"
    )
    safety_critical: bool = Field(
        description="True if symptom could indicate a life-threatening condition"
    )


class EnrichmentBatch(BaseModel):
    symptoms: list[EnrichmentEntry]


class CatalogEnricher:
    def __init__(self, provider: LLMProvider, config: PipelineConfig):
        self.provider = provider
        self.config = config

    async def enrich_batch(
        self,
        batch: list[EnrichedSymptom],
        all_symptoms: list[EnrichedSymptom],
    ) -> list[EnrichmentEntry]:
        """Enrich a batch of symptoms."""
        symptoms_text = []
        for s in batch:
            neighbors = [
                n for n in all_symptoms
                if n.code != s.code and n.triage_depth == s.triage_depth
            ][:5]
            neighbor_labels = ", ".join(n.label_it for n in neighbors)

            symptoms_text.append(
                f"Code: {s.code}\n"
                f"Label (IT): {s.label_it}\n"
                f"Label (EN): {s.label_en}\n"
                f"Definition: {s.short_definition}\n"
                f"Similar symptoms for disambiguation: {neighbor_labels}"
            )

        user_message = (
            "Enrich the following symptoms:\n\n"
            + "\n\n---\n\n".join(symptoms_text)
        )

        response = await self.provider.extract_structured(
            system_prompt=ENRICHMENT_SYSTEM_PROMPT,
            user_message=user_message,
            response_schema=EnrichmentBatch,
            model=self.config.frontier_model,
            temperature=0.3,
            cache_system_prompt=False,
        )

        if response.parsed:
            return response.parsed.symptoms
        return []

    async def enrich_all(
        self,
        catalog: list[EnrichedSymptom],
        batch_size: int = 5,
    ) -> list[EnrichedSymptom]:
        """Enrich all symptoms and save to disk."""
        enriched = []
        enrichment_map: dict[str, EnrichmentEntry] = {}

        batches = [
            catalog[i : i + batch_size]
            for i in range(0, len(catalog), batch_size)
        ]

        logger.info(f"Enriching {len(catalog)} symptoms in {len(batches)} batches")

        for i, batch in enumerate(batches):
            logger.info(f"  Batch {i + 1}/{len(batches)}: {[s.label_it for s in batch]}")
            try:
                entries = await self.enrich_batch(batch, catalog)
                for entry in entries:
                    enrichment_map[entry.code] = entry
            except Exception as e:
                logger.error(f"  Error enriching batch {i + 1}: {e}")

        for s in catalog:
            entry = enrichment_map.get(s.code)
            if entry:
                enriched.append(
                    EnrichedSymptom(
                        code=s.code,
                        label_it=s.label_it,
                        label_en=s.label_en,
                        triage_depth=s.triage_depth,
                        short_definition=s.short_definition,
                        examples_it=entry.examples_it,
                        synonyms_it=entry.synonyms_it,
                        negation_patterns_it=entry.negation_patterns_it,
                        disambiguation=entry.disambiguation,
                        safety_critical=entry.safety_critical,
                    )
                )
            else:
                enriched.append(s)

        data = [s.model_dump() for s in enriched]
        OUTPUT_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Enriched catalog saved to {OUTPUT_PATH}")
        return enriched
