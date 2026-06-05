"""Batch evaluation runner over the test dataset."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from ..catalog.loader import build_label_maps
from ..models import EnrichedSymptom, GatedResult, PipelineConfig
from ..pipeline.orchestrator import PipelineOrchestrator
from .metrics import (
    CaseMetrics,
    compute_aggregate_metrics,
    compute_case_metrics,
    compute_symptom_metrics,
)

logger = logging.getLogger(__name__)

ALIASES: dict[str, str] = {
    "poliuria": "Poliuria (emissione di abbondante quantità di urina)",
    "pollachiuria": "Pollachiuria (necessità di urinare molto spesso ma con piccole quantità)",
    "naso ostruito e che cola": "Naso ostruito o che cola",
    "sincope o collasso": "Sincope, collasso",
    "sincope": "Sincope, collasso",
    "collasso": "Sincope, collasso",
    "vertigini": "Vertigini, capogiro",
    "capogiro": "Vertigini, capogiro",
    "dolore muscolare": "Dolore muscolare/Scheletrico",
    "dolore muscolare scheletrico": "Dolore muscolare/Scheletrico",
    "dolore muscolare e scheletrico": "Dolore muscolare/Scheletrico",
    "dolore muscolare e/o scheletrico": "Dolore muscolare/Scheletrico",
    "scheletrico": "Dolore muscolare/Scheletrico",
    "sincope/collasso": "Sincope, collasso",
    "irrequietezza/pianto inconsolabile": "Irrequietezza o pianto inconsolabile",
    "naso ostruito e/o che cola": "Naso ostruito o che cola",
    "prurito o bruciore anale/genitale": "Prurito o bruciore anale o delle aree genitali",
    "prurito/bruciore anale o genitale": "Prurito o bruciore anale o delle aree genitali",
    "arrossamento o gonfiore anale/genitale": "Arrossamento o gonfiore anale o delle aree genitali",
    "arrossamento/gonfiore anale o genitale": "Arrossamento o gonfiore anale o delle aree genitali",
    "prurito o bruciore anale o genitale": "Prurito o bruciore anale o delle aree genitali",
    "prurito genitale": "Prurito o bruciore anale o delle aree genitali",
    "prurito o bruciore anale": "Prurito o bruciore anale o delle aree genitali",
    "arrossamento o gonfiore anale o genitale": "Arrossamento o gonfiore anale o delle aree genitali",
    "arrossamento genitale": "Arrossamento o gonfiore anale o delle aree genitali",
    "arrossamento o gonfiore anale": "Arrossamento o gonfiore anale o delle aree genitali",
    "ingestione di corpo estraneo": "Ingestione di corpi estranei",
    "russamento": "Russamento nel sonno",
    "irrequietezza": "Irrequietezza o pianto inconsolabile",
    "pianto inconsolabile": "Irrequietezza o pianto inconsolabile",
    "tosse": "Tosse",
    "tossa": "Tosse",
    "zoppia": "Zoppìa",
    "ingestione di corpi estranei": "Ingestione di corpo estraneo",
}


@dataclass
class CaseResult:
    case_id: str
    message: str
    expected: list[str]
    predicted: list[str]
    metrics: CaseMetrics
    case_type: str = ""
    section: str = ""
    error: str | None = None
    result: GatedResult | None = None


@dataclass
class EvaluationReport:
    case_results: list[CaseResult] = field(default_factory=list)
    per_symptom: dict[str, object] = field(default_factory=dict)
    aggregate: object = None
    config: dict = field(default_factory=dict)


class EvaluationRunner:
    def __init__(
        self,
        pipeline: PipelineOrchestrator,
        config: PipelineConfig,
        catalog: list[EnrichedSymptom],
    ):
        self.pipeline = pipeline
        self.config = config
        self.catalog = catalog
        self._code_to_label, self._label_to_code = build_label_maps(catalog)
        self._canon = set(self._code_to_label.values())
        self._canon_lower = {label.lower(): label for label in self._canon}

    def _canonicalize(self, label: str) -> str | None:
        if label in self._canon:
            return label
        low = label.lower().strip().strip(".")
        if low in self._canon_lower:
            return self._canon_lower[low]
        if low in ALIASES:
            return ALIASES[low]
        return None

    def _predicted_labels(self, result: GatedResult) -> list[str]:
        labels = []
        seen = set()
        for s in result.confirmed + result.flagged_for_review:
            if s.label_it not in seen:
                labels.append(s.label_it)
                seen.add(s.label_it)
        return labels

    async def _evaluate_case(
        self, case: dict, semaphore: asyncio.Semaphore
    ) -> CaseResult:
        case_id = case.get("id", "?")
        message = case.get("message", "")
        raw_expected = case.get("expected_symptoms_canonical", [])
        case_type = case.get("case_type", "")
        section = case.get("section", "")
        negated_symptom_raw = case.get("negated_symptom") or case.get("resolved_symptom")

        expected = []
        for label in raw_expected:
            canon = self._canonicalize(label)
            if canon:
                expected.append(canon)

        negated_symptom = None
        if negated_symptom_raw and case_type in ("negation", "past_resolved"):
            negated_symptom = self._canonicalize(negated_symptom_raw) or negated_symptom_raw

        async with semaphore:
            try:
                result = await self.pipeline.run(message)
                predicted = self._predicted_labels(result)

                canon_predicted = []
                for label in predicted:
                    canon = self._canonicalize(label)
                    if canon:
                        canon_predicted.append(canon)
                    else:
                        canon_predicted.append(label)

                metrics = compute_case_metrics(
                    canon_predicted, expected, negated_symptom=negated_symptom
                )

                return CaseResult(
                    case_id=case_id,
                    message=message,
                    expected=expected,
                    predicted=canon_predicted,
                    metrics=metrics,
                    case_type=case_type,
                    section=section,
                    result=result,
                )
            except Exception as e:
                logger.error(f"Error evaluating case {case_id}: {e}")
                return CaseResult(
                    case_id=case_id,
                    message=message,
                    expected=expected,
                    predicted=[],
                    metrics=compute_case_metrics([], expected),
                    case_type=case_type,
                    section=section,
                    error=str(e),
                )

    async def run_dataset(
        self,
        dataset_path: str | Path = "data/eval/test_dataset.jsonl",
        max_cases: int | None = None,
        case_types: list[str] | None = None,
    ) -> EvaluationReport:
        path = Path(dataset_path)
        cases = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                if case_types and case.get("case_type", "") not in case_types:
                    has_canonical = bool(case.get("expected_symptoms_canonical"))
                    if case_types and not case.get("case_type"):
                        pass
                    else:
                        continue
                canonical = case.get("expected_symptoms_canonical", [])
                unresolved = case.get("expected_symptoms_unresolved", [])
                if unresolved and not canonical:
                    continue
                cases.append(case)

        if max_cases:
            cases = cases[:max_cases]

        logger.info(f"Running evaluation on {len(cases)} cases (mode={self.config.mode})")

        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        tasks = [self._evaluate_case(case, semaphore) for case in cases]

        results = []
        with tqdm(total=len(tasks), desc="Evaluating") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        results.sort(key=lambda r: r.case_id)

        all_predictions = [r.predicted for r in results]
        all_golds = [r.expected for r in results]

        per_symptom = compute_symptom_metrics(all_predictions, all_golds, self.catalog)
        aggregate = compute_aggregate_metrics(
            per_symptom,
            recall_floor=self.config.recall_floor,
            min_support=self.config.min_support_for_floor,
            total_cases=len(results),
        )

        return EvaluationReport(
            case_results=results,
            per_symptom=per_symptom,
            aggregate=aggregate,
            config=self.config.model_dump(),
        )
