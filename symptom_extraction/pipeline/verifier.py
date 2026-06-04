"""Verification pass — double-check each extracted symptom against evidence."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from ..llm.base import LLMProvider, LLMUsage
from ..models import ExtractedSymptom, ExcludedSymptom, PipelineConfig

logger = logging.getLogger(__name__)


class VerificationItem(BaseModel):
    code: str
    label_it: str
    verdict: Literal["keep", "remove", "downgrade"] = Field(
        description="keep=symptom is genuinely present; "
        "remove=symptom is not actually present or is a consequence of another; "
        "downgrade=uncertain, move to excluded"
    )
    reason: str = Field(
        description="Brief explanation of the verdict"
    )


class VerificationResult(BaseModel):
    items: list[VerificationItem]


VERIFY_SYSTEM_PROMPT = """\
You are a clinical verification system. You receive a parent's message and \
a list of symptoms that were extracted from it. Your job is to verify each \
symptom and decide if it should be kept, removed, or downgraded.

## Verification rules
For each symptom, decide:
- "keep": The symptom IS genuinely and independently described in the message.
- "remove": The symptom is NOT actually present, OR it is merely a \
consequence/description of another extracted symptom (not independently described), \
OR it was over-extracted from ambiguous evidence.
- "downgrade": The evidence is very weak or ambiguous.

## Key checks
1. Is the evidence_span actually describing THIS specific symptom, or something adjacent?
2. Is this symptom independently described, or just a logical consequence of another symptom?
   - Example: if "dolore al ginocchio" is extracted AND "zoppia" is extracted, \
but the message only says "gli fa male il ginocchio e cammina male" → \
remove zoppia (the limping is just a consequence of knee pain).
3. Does the evidence actually support a CURRENT symptom, or is it negated/resolved?
4. Is the symptom the MOST SPECIFIC match? If a more specific symptom is also extracted, \
the broader one should be removed.
"""


class SymptomVerifier:
    def __init__(self, provider: LLMProvider, config: PipelineConfig):
        self._provider = provider
        self._config = config

    async def verify(
        self,
        message: str,
        symptoms: list[ExtractedSymptom],
    ) -> tuple[list[ExtractedSymptom], list[ExcludedSymptom], LLMUsage]:
        if not symptoms:
            return [], [], LLMUsage()

        symptoms_text = "\n".join(
            f"- {s.code} {s.label_it}: evidence=\"{s.evidence_span}\" "
            f"confidence={s.confidence.value} hedged={s.hedged}"
            for s in symptoms
        )

        user_msg = (
            f"Parent message: \"{message}\"\n\n"
            f"Extracted symptoms to verify:\n{symptoms_text}\n\n"
            f"Verify each symptom."
        )

        response = await self._provider.extract_structured(
            system_prompt=VERIFY_SYSTEM_PROMPT,
            user_message=user_msg,
            response_schema=VerificationResult,
            model=self._config.frontier_model,
            temperature=0.0,
            cache_system_prompt=False,
        )

        if response.parsed is None:
            return symptoms, [], response.usage

        verdict_map = {item.code: item for item in response.parsed.items}

        kept = []
        removed = []
        for s in symptoms:
            v = verdict_map.get(s.code)
            if v is None or v.verdict == "keep":
                kept.append(s)
            elif v.verdict == "remove":
                removed.append(ExcludedSymptom(
                    code=s.code,
                    label_it=s.label_it,
                    reason="below_threshold",
                    evidence_span=s.evidence_span,
                ))
            elif v.verdict == "downgrade":
                removed.append(ExcludedSymptom(
                    code=s.code,
                    label_it=s.label_it,
                    reason="below_threshold",
                    evidence_span=s.evidence_span,
                ))

        return kept, removed, response.usage
