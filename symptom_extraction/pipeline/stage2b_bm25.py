"""Stage 2b: BM25 lexical retrieval with Italian lemmatization."""

from __future__ import annotations

import logging

import simplemma
from rank_bm25 import BM25Okapi

from ..models import EnrichedSymptom

logger = logging.getLogger(__name__)


def _lemmatize_it(text: str) -> list[str]:
    """Tokenize and lemmatize Italian text."""
    tokens = text.lower().split()
    cleaned = []
    for t in tokens:
        t = t.strip(".,;:!?()\"'")
        if len(t) > 1:
            lemma = simplemma.lemmatize(t, lang="it")
            cleaned.append(lemma)
    return cleaned


class BM25Retriever:
    def __init__(self, catalog: list[EnrichedSymptom]):
        self._codes: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._build_index(catalog)

    def _build_index(self, catalog: list[EnrichedSymptom]) -> None:
        corpus = []
        codes = []
        for s in catalog:
            tokens = set()
            tokens.update(_lemmatize_it(s.label_it))
            if s.short_definition:
                tokens.update(_lemmatize_it(s.short_definition))
            for ex in s.examples_it:
                tokens.update(_lemmatize_it(ex))
            for syn in s.synonyms_it:
                tokens.update(_lemmatize_it(syn))
            corpus.append(list(tokens))
            codes.append(s.code)

        self._codes = codes
        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built: {len(codes)} symptoms")

    def query(
        self, clause_text: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Return top-k (code, score) pairs by BM25 ranking."""
        query_tokens = _lemmatize_it(clause_text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        indexed = [(self._codes[i], float(scores[i])) for i in range(len(scores))]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]
