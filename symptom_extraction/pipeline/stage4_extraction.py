"""Stage 4: Constrained final extraction — frontier LLM with enum schema."""

from __future__ import annotations

import logging

from ..llm.base import LLMProvider, LLMUsage
from ..models import (
    EnrichedSymptom,
    ExcludedSymptom,
    ExtractedSymptom,
    ExtractionResult,
    PipelineConfig,
    build_extraction_model,
)
from ..prompts.templates import build_stage4_prompt

logger = logging.getLogger(__name__)


class ConstrainedExtractor:
    def __init__(self, provider: LLMProvider, config: PipelineConfig):
        self._provider = provider
        self._config = config

    async def extract(
        self,
        message: str,
        candidates: list[EnrichedSymptom],
    ) -> tuple[ExtractionResult, LLMUsage]:
        """Run constrained extraction on the original message with top-K candidates."""
        if not candidates:
            return ExtractionResult(), LLMUsage()

        allowed_labels = [(s.code, s.label_it) for s in candidates]
        response_model = build_extraction_model(allowed_labels)

        system_prompt = build_stage4_prompt(candidates)

        response = await self._provider.extract_structured(
            system_prompt=system_prompt,
            user_message=message,
            response_schema=response_model,
            model=self._config.frontier_model,
            temperature=self._config.temperature,
            cache_system_prompt=False,
        )

        if response.parsed is None:
            logger.warning("Stage 4 returned no parsed result")
            return ExtractionResult(), response.usage

        raw = response.parsed
        symptoms = []
        excluded = []

        for s in raw.symptoms:
            if s.negated or s.temporal_status.value == "past_resolved":
                excluded.append(ExcludedSymptom(
                    code=s.code,
                    label_it=s.label_it,
                    reason="negated" if s.negated else "past_resolved",
                    evidence_span=s.evidence_span,
                ))
            else:
                symptoms.append(ExtractedSymptom(
                    code=s.code,
                    label_it=s.label_it,
                    evidence_span=s.evidence_span,
                    negated=s.negated,
                    hedged=s.hedged,
                    temporal_status=s.temporal_status,
                    confidence=s.confidence,
                    onset=s.onset,
                ))

        for e in raw.excluded:
            excluded.append(ExcludedSymptom(
                code=e.code,
                label_it=e.label_it,
                reason=e.reason,
                evidence_span=e.evidence_span,
            ))

        return ExtractionResult(symptoms=symptoms, excluded=excluded), response.usage
