"""Pre-computed symptom vector index — build once, query fast."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..models import EnrichedSymptom, PipelineConfig
from .encoder import compute_cosine_similarity, encode_texts

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class SymptomVectorIndex:
    def __init__(
        self,
        catalog: list[EnrichedSymptom],
        config: PipelineConfig | None = None,
    ):
        self.catalog = catalog
        self.model_name = (
            config.embedding_model
            if config
            else "intfloat/multilingual-e5-large-instruct"
        )
        self._codes: list[str] = [s.code for s in catalog]
        self._vectors: np.ndarray | None = None
        self._cache_path = CACHE_DIR / "symptom_vectors.npz"

    def _build_document_texts(self) -> list[str]:
        """Build the text representation for each symptom to embed."""
        texts = []
        for s in self.catalog:
            parts = [s.label_it]
            if s.short_definition:
                parts.append(s.short_definition)
            if s.examples_it:
                parts.append(" | ".join(s.examples_it[:5]))
            if s.synonyms_it:
                parts.append(", ".join(s.synonyms_it[:5]))
            texts.append("passage: " + ". ".join(parts))
        return texts

    def build_and_save(self) -> None:
        """Build vectors from the catalog and save to disk."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        texts = self._build_document_texts()
        logger.info(f"Encoding {len(texts)} symptom documents...")
        vectors = encode_texts(texts, self.model_name)
        self._vectors = vectors

        np.savez_compressed(
            self._cache_path,
            vectors=vectors,
            codes=np.array(self._codes),
        )
        logger.info(f"Saved symptom vectors to {self._cache_path}")

    def load(self) -> bool:
        """Load pre-computed vectors from disk. Returns True if successful."""
        if self._cache_path.exists():
            data = np.load(self._cache_path, allow_pickle=True)
            self._vectors = data["vectors"]
            saved_codes = list(data["codes"])
            if saved_codes == self._codes:
                logger.info(f"Loaded {len(self._codes)} symptom vectors from cache")
                return True
            logger.warning("Cache codes mismatch, will recompute")
        return False

    def ensure_loaded(self) -> None:
        """Load from cache or build from scratch."""
        if self._vectors is not None:
            return
        if not self.load():
            self.build_and_save()

    def query(
        self, text: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find the top-k most similar symptoms for a query text.

        Returns list of (symptom_code, score) tuples.
        """
        self.ensure_loaded()

        query_vec = encode_texts(
            [f"query: {text}"], self.model_name
        )[0]

        scores = compute_cosine_similarity(query_vec, self._vectors)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._codes[i], float(scores[i]))
            for i in top_indices
        ]
