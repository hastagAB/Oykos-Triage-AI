"""Stage 3: Reciprocal Rank Fusion — merge retrieval results."""

from __future__ import annotations

from collections import defaultdict

from ..models import CandidateSymptom, EnrichedSymptom


class RRFFusion:
    def __init__(self, k_constant: int = 60, top_k: int = 15):
        self.k_constant = k_constant
        self.top_k = top_k

    def fuse(
        self,
        dense_lists: list[list[tuple[str, float]]],
        bm25_lists: list[list[tuple[str, float]]],
        llm_lists: list[list[tuple[str, float]]],
        clause_texts: list[str],
        catalog_by_code: dict[str, EnrichedSymptom],
    ) -> list[CandidateSymptom]:
        """Fuse retrieval results across clauses and retrievers.

        For each candidate symptom s:
            score(s) = sum over all (clause, retriever) pairs of
                       1 / (k_constant + rank_in_that_list(s))
        """
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        clause_sources: dict[str, set[str]] = defaultdict(set)

        all_lists = [
            ("dense", dense_lists),
            ("bm25", bm25_lists),
            ("llm_extract", llm_lists),
        ]

        for retriever_name, per_clause_lists in all_lists:
            for clause_idx, ranked_list in enumerate(per_clause_lists):
                clause_text = (
                    clause_texts[clause_idx] if clause_idx < len(clause_texts) else ""
                )
                for rank, (code, _raw_score) in enumerate(ranked_list):
                    rrf_contribution = 1.0 / (self.k_constant + rank + 1)
                    scores[code] += rrf_contribution
                    sources[code].add(retriever_name)
                    clause_sources[code].add(clause_text)

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        candidates = []
        for code, score in sorted_candidates[: self.top_k]:
            symptom = catalog_by_code.get(code)
            label_it = symptom.label_it if symptom else code
            candidates.append(
                CandidateSymptom(
                    code=code,
                    label_it=label_it,
                    rrf_score=score,
                    source_retrievers=sorted(sources[code]),
                    source_clauses=sorted(clause_sources[code]),
                )
            )

        return candidates
