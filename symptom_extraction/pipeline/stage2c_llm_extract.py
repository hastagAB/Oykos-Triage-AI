"""Stage 2c: LLM free-extraction followed by nearest-neighbor lookup."""

from __future__ import annotations

import json
import logging

from ..llm.base import LLMProvider, LLMUsage
from ..models import EnrichedSymptom, PipelineConfig
from ..prompts.templates import STAGE2C_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class LLMExtractRetriever:
    def __init__(
        self,
        catalog: list[EnrichedSymptom],
        provider: LLMProvider,
        config: PipelineConfig,
        dense_retriever,
    ):
        self._catalog = catalog
        self._provider = provider
        self._config = config
        self._dense = dense_retriever
        self.last_usage = LLMUsage()

    async def query(
        self, clause_text: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Extract symptom phrases via LLM, then map to catalog via embeddings."""
        self.last_usage = LLMUsage()

        prompt = STAGE2C_EXTRACTION_PROMPT.format(message=clause_text)

        try:
            response = await self._provider.extract_text(
                system_prompt="You extract medical symptoms from Italian text. Always return valid JSON.",
                user_message=prompt,
                model=self._config.cheap_model,
                temperature=0.0,
                max_tokens=1024,
            )
            self.last_usage = response.usage

            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]

            phrases = json.loads(text)
        except Exception as e:
            logger.warning(f"Stage 2c LLM extraction failed: {e}")
            return []

        all_results: dict[str, float] = {}
        for item in phrases:
            phrase = item.get("phrase", "") if isinstance(item, dict) else str(item)
            if not phrase:
                continue

            nn_results = await self._dense.query(phrase, top_k=5)
            for code, score in nn_results:
                if code in all_results:
                    all_results[code] = max(all_results[code], score)
                else:
                    all_results[code] = score

        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
