"""Stage 2a: Dense embedding retrieval."""

from __future__ import annotations

from ..embeddings.index import SymptomVectorIndex
from ..models import EnrichedSymptom, PipelineConfig


class DenseRetriever:
    def __init__(self, catalog: list[EnrichedSymptom], config: PipelineConfig):
        self._index = SymptomVectorIndex(catalog, config)

    async def query(
        self, clause_text: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Return top-k (code, score) pairs by dense similarity."""
        return self._index.query(clause_text, top_k=top_k)
