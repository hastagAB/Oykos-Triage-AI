"""Pipeline orchestrator — runs baseline or full 5-stage pipeline."""

from __future__ import annotations

import logging
import time

from ..catalog.loader import build_label_maps
from ..llm.base import LLMProvider, LLMUsage
from ..models import (
    CandidateSymptom,
    EnrichedSymptom,
    ExtractionResult,
    ExtractedSymptom,
    ExcludedSymptom,
    GatedResult,
    PipelineConfig,
    PipelineMetadata,
    build_extraction_model,
)
from ..prompts.templates import build_baseline_prompt, build_stage4_prompt

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        config: PipelineConfig,
        provider: LLMProvider,
        catalog: list[EnrichedSymptom],
    ):
        self.config = config
        self.provider = provider
        self.catalog = catalog
        self._code_to_label, self._label_to_code = build_label_maps(catalog)
        self._catalog_by_code = {s.code: s for s in catalog}

        self._baseline_prompt = build_baseline_prompt(catalog)
        self._all_labels = [(s.code, s.label_it) for s in catalog]

        # Pipeline stages (initialized lazily)
        self._segmenter = None
        self._dense_retriever = None
        self._bm25_retriever = None
        self._llm_extractor = None
        self._fusion = None
        self._constrained_extractor = None
        self._confidence_gate = None

    async def run(self, message: str) -> GatedResult:
        if self.config.mode == "pipeline":
            return await self.run_pipeline(message)
        return await self.run_baseline(message)

    async def run_baseline(self, message: str) -> GatedResult:
        start = time.perf_counter()

        response_model = build_extraction_model(self._all_labels)
        response = await self.provider.extract_structured(
            system_prompt=self._baseline_prompt,
            user_message=message,
            response_schema=response_model,
            model=self.config.frontier_model,
            temperature=self.config.temperature,
            cache_system_prompt=self.config.cache_system_prompt,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.parsed is None:
            return GatedResult(
                pipeline_metadata=PipelineMetadata(
                    mode="baseline",
                    latency_ms=elapsed_ms,
                    llm_calls=1,
                    total_input_tokens=response.usage.input_tokens,
                    total_output_tokens=response.usage.output_tokens,
                ),
            )

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

        total_llm_calls = 1
        total_input = response.usage.input_tokens
        total_output = response.usage.output_tokens

        if self.config.use_verifier and len(symptoms) > 1:
            from .verifier import SymptomVerifier
            verifier = SymptomVerifier(self.provider, self.config)
            kept, removed, verify_usage = await verifier.verify(message, symptoms)
            symptoms = kept
            excluded.extend(removed)
            total_llm_calls += verify_usage.calls
            total_input += verify_usage.input_tokens
            total_output += verify_usage.output_tokens

        elapsed_ms = (time.perf_counter() - start) * 1000

        return GatedResult(
            confirmed=symptoms,
            flagged_for_review=[],
            abstained=[],
            excluded=excluded,
            pipeline_metadata=PipelineMetadata(
                mode="baseline",
                latency_ms=elapsed_ms,
                llm_calls=total_llm_calls,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
            ),
        )

    async def run_pipeline(self, message: str) -> GatedResult:
        import asyncio
        start = time.perf_counter()
        total_usage = LLMUsage()

        # Stage 1: Clause segmentation
        from .stage1_segmentation import RuleBasedSegmenter
        if self._segmenter is None:
            self._segmenter = RuleBasedSegmenter()
        clauses = self._segmenter.segment(message)
        clause_texts = [c.text for c in clauses]

        # Stage 2: Parallel retrieval
        from .stage2a_dense import DenseRetriever
        from .stage2b_bm25 import BM25Retriever
        from .stage2c_llm_extract import LLMExtractRetriever

        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever(self.catalog, self.config)
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(self.catalog)
        if self._llm_extractor is None:
            self._llm_extractor = LLMExtractRetriever(
                self.catalog, self.provider, self.config, self._dense_retriever
            )

        all_dense = []
        all_bm25 = []
        all_llm = []

        for clause_text in clause_texts:
            bm25_results = self._bm25_retriever.query(clause_text)

            dense_results, llm_results = await asyncio.gather(
                self._dense_retriever.query(clause_text),
                self._llm_extractor.query(clause_text),
            )

            all_dense.append(dense_results)
            all_bm25.append(bm25_results)
            all_llm.append(llm_results)

            total_usage.merge(self._llm_extractor.last_usage)

        # Stage 3: RRF fusion
        from .stage3_fusion import RRFFusion
        if self._fusion is None:
            self._fusion = RRFFusion(
                k_constant=self.config.rrf_k_constant,
                top_k=self.config.top_k_candidates,
            )

        candidates = self._fusion.fuse(
            dense_lists=all_dense,
            bm25_lists=all_bm25,
            llm_lists=all_llm,
            clause_texts=clause_texts,
            catalog_by_code=self._catalog_by_code,
        )

        # Stage 4: Constrained extraction
        from .stage4_extraction import ConstrainedExtractor
        if self._constrained_extractor is None:
            self._constrained_extractor = ConstrainedExtractor(
                self.provider, self.config
            )

        candidate_symptoms = [
            self._catalog_by_code[c.code]
            for c in candidates
            if c.code in self._catalog_by_code
        ]

        extraction, extract_usage = await self._constrained_extractor.extract(
            message=message,
            candidates=candidate_symptoms,
        )
        total_usage.merge(extract_usage)

        # Stage 5: Confidence gating
        from .stage5_gating import ConfidenceGate
        if self._confidence_gate is None:
            self._confidence_gate = ConfidenceGate(
                high_threshold=self.config.high_confidence_threshold,
                review_threshold=self.config.review_confidence_threshold,
            )

        candidate_score_map = {c.code: c for c in candidates}
        gated = self._confidence_gate.gate(extraction, candidate_score_map)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return GatedResult(
            confirmed=gated.confirmed,
            flagged_for_review=gated.flagged_for_review,
            abstained=gated.abstained,
            excluded=gated.excluded,
            pipeline_metadata=PipelineMetadata(
                mode="pipeline",
                clauses=clause_texts,
                candidate_count=len(candidates),
                latency_ms=elapsed_ms,
                llm_calls=total_usage.calls,
                total_input_tokens=total_usage.input_tokens,
                total_output_tokens=total_usage.output_tokens,
            ),
        )
