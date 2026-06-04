"""Stage 5: Confidence gating and abstention — rule-based."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    CandidateSymptom,
    ConfidenceLevel,
    ExcludedSymptom,
    ExtractedSymptom,
    ExtractionResult,
)

CONFIDENCE_SCORES = {
    ConfidenceLevel.HIGH: 1.0,
    ConfidenceLevel.MEDIUM: 0.6,
    ConfidenceLevel.LOW: 0.3,
}

RETRIEVER_AGREEMENT_BONUS = 0.15
TOP_RANK_BONUS = 0.1
TOP_RANK_CUTOFF = 5


@dataclass
class GatingResult:
    confirmed: list[ExtractedSymptom]
    flagged_for_review: list[ExtractedSymptom]
    abstained: list[ExtractedSymptom]
    excluded: list[ExcludedSymptom]


class ConfidenceGate:
    def __init__(
        self,
        high_threshold: float = 0.7,
        review_threshold: float = 0.4,
    ):
        self.high_threshold = high_threshold
        self.review_threshold = review_threshold

    def _compute_score(
        self,
        symptom: ExtractedSymptom,
        candidate: CandidateSymptom | None,
    ) -> float:
        base = CONFIDENCE_SCORES.get(symptom.confidence, 0.5)

        if candidate:
            n_retrievers = len(candidate.source_retrievers)
            agreement_bonus = min(
                (n_retrievers - 1) * RETRIEVER_AGREEMENT_BONUS,
                2 * RETRIEVER_AGREEMENT_BONUS,
            )
            base += agreement_bonus

            if candidate.rrf_score > 0:
                base += TOP_RANK_BONUS

        return min(base, 1.0)

    def gate(
        self,
        extraction: ExtractionResult,
        candidate_scores: dict[str, CandidateSymptom],
    ) -> GatingResult:
        confirmed = []
        flagged = []
        abstained = []

        for symptom in extraction.symptoms:
            candidate = candidate_scores.get(symptom.code)
            score = self._compute_score(symptom, candidate)

            if score >= self.high_threshold:
                confirmed.append(symptom)
            elif score >= self.review_threshold:
                flagged.append(symptom)
            else:
                abstained.append(symptom)

        return GatingResult(
            confirmed=confirmed,
            flagged_for_review=flagged,
            abstained=abstained,
            excluded=extraction.excluded,
        )
