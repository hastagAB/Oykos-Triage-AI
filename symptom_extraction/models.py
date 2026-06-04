"""Pydantic models — the shared vocabulary for the entire pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TemporalStatus(str, Enum):
    CURRENT = "current"
    PAST_RESOLVED = "past_resolved"
    CHRONIC = "chronic"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Extraction output
# ---------------------------------------------------------------------------

class ExtractedSymptom(BaseModel):
    code: str = Field(description="Canonical symptom code, e.g. SI001")
    label_it: str = Field(description="Italian label from catalog")
    evidence_span: str = Field(description="Verbatim substring from parent message")
    negated: bool = Field(default=False)
    hedged: bool = Field(default=False)
    temporal_status: TemporalStatus = Field(default=TemporalStatus.CURRENT)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
    onset: str | None = Field(default=None, description="Temporal onset if mentioned")


class ExcludedSymptom(BaseModel):
    code: str
    label_it: str
    reason: Literal["negated", "past_resolved", "below_threshold"]
    evidence_span: str | None = None


class ExtractionResult(BaseModel):
    symptoms: list[ExtractedSymptom] = Field(default_factory=list)
    excluded: list[ExcludedSymptom] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------------

class PipelineMetadata(BaseModel):
    mode: Literal["baseline", "pipeline"]
    clauses: list[str] | None = None
    candidate_count: int | None = None
    retriever_scores: dict[str, list[tuple[str, float]]] | None = None
    latency_ms: float = 0.0
    llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class GatedResult(BaseModel):
    confirmed: list[ExtractedSymptom] = Field(default_factory=list)
    flagged_for_review: list[ExtractedSymptom] = Field(default_factory=list)
    abstained: list[ExtractedSymptom] = Field(default_factory=list)
    excluded: list[ExcludedSymptom] = Field(default_factory=list)
    pipeline_metadata: PipelineMetadata


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class EnrichedSymptom(BaseModel):
    code: str
    label_it: str
    label_en: str
    triage_depth: str
    short_definition: str
    examples_it: list[str] = Field(default_factory=list)
    synonyms_it: list[str] = Field(default_factory=list)
    negation_patterns_it: list[str] = Field(default_factory=list)
    disambiguation: str = ""
    safety_critical: bool = False


# ---------------------------------------------------------------------------
# Clause segmentation
# ---------------------------------------------------------------------------

class Clause(BaseModel):
    text: str
    has_negation: bool = False
    temporal_marker: str | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class CandidateSymptom(BaseModel):
    code: str
    label_it: str
    rrf_score: float = 0.0
    source_retrievers: list[str] = Field(default_factory=list)
    source_clauses: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class CaseMetrics(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    predicted: list[str] = Field(default_factory=list)
    expected: list[str] = Field(default_factory=list)


class SymptomMetrics(BaseModel):
    code: str
    label_it: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


class AggregateMetrics(BaseModel):
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    total_cases: int = 0
    symptoms_below_recall_floor: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    mode: Literal["baseline", "pipeline"] = "baseline"
    provider: Literal["anthropic", "openai", "gemini"] = "openai"
    frontier_model: str = "claude-sonnet-4-20250514"
    cheap_model: str = "claude-3-5-haiku-20241022"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    top_k_candidates: int = 15
    rrf_k_constant: int = 60
    high_confidence_threshold: float = 0.7
    review_confidence_threshold: float = 0.4
    temperature: float = 0.0
    cache_system_prompt: bool = True
    use_few_shot: bool = False
    use_verifier: bool = True
    recall_floor: float = 0.85
    min_support_for_floor: int = 8
    max_concurrency: int = 5


# ---------------------------------------------------------------------------
# Dynamic enum model builder
# ---------------------------------------------------------------------------

def build_extraction_model(
    allowed_labels: list[tuple[str, str]],
) -> type[BaseModel]:
    """Build a Pydantic model with label_it constrained to a Literal enum.

    Args:
        allowed_labels: list of (code, label_it) tuples.

    Returns:
        A Pydantic model class whose label_it is enum-constrained.
    """
    codes = [code for code, _ in allowed_labels]
    labels = [label for _, label in allowed_labels]

    code_literal = Literal[tuple(codes)]  # type: ignore[valid-type]
    label_literal = Literal[tuple(labels)]  # type: ignore[valid-type]

    class ConstrainedSymptom(BaseModel):
        code: code_literal = Field(description="Symptom code from the candidate list")  # type: ignore[valid-type]
        label_it: label_literal = Field(description="Italian label from the candidate list")  # type: ignore[valid-type]
        evidence_span: str = Field(description="Verbatim substring from parent message")
        negated: bool = Field(default=False)
        hedged: bool = Field(default=False)
        temporal_status: TemporalStatus = Field(default=TemporalStatus.CURRENT)
        confidence: ConfidenceLevel = Field(default=ConfidenceLevel.HIGH)
        onset: str | None = Field(default=None)

    class ConstrainedExcluded(BaseModel):
        code: code_literal = Field(description="Symptom code")  # type: ignore[valid-type]
        label_it: label_literal = Field(description="Italian label")  # type: ignore[valid-type]
        reason: Literal["negated", "past_resolved", "below_threshold"] = Field(
            description="Why this symptom was excluded"
        )
        evidence_span: str | None = Field(default=None)

    class ConstrainedExtractionResult(BaseModel):
        symptoms: list[ConstrainedSymptom] = Field(
            default_factory=list,
            description="Symptoms that are currently present and actively asserted",
        )
        excluded: list[ConstrainedExcluded] = Field(
            default_factory=list,
            description="Symptoms that were considered but excluded (negated or past-resolved)",
        )

    return ConstrainedExtractionResult
