"""Prompt templates for all pipeline stages."""

from __future__ import annotations

from ..models import EnrichedSymptom


# ---------------------------------------------------------------------------
# Baseline: single LLM call with full catalog
# ---------------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """\
You are a pediatric symptom extractor. You analyze messages written in Italian \
by parents describing their child's symptoms to a pediatrician.

## Your task
Extract EVERY currently present symptom from the message. Map each to exactly \
one canonical symptom from the catalog below. A single message may contain \
zero, one, or many symptoms.

## Critical rules
1. Use ONLY symptoms from the catalog below. Never invent labels.
2. Negated symptoms are EXCLUDED: "non ha la febbre" means fever is ABSENT. \
Do NOT include it in symptoms. Place it in excluded with reason "negated".
3. Past-resolved symptoms are EXCLUDED: "la febbre è passata", \
"la tosse di settimana scorsa è finita" mean the symptom is RESOLVED. \
Do NOT include it in symptoms. Place it in excluded with reason "past_resolved".
4. Hedged/uncertain symptoms ARE INCLUDED with hedged=true: \
"credo gli faccia male la pancia", "forse ha la febbre" → include with hedged=true, \
confidence="medium".
5. evidence_span MUST be a verbatim substring of the parent's message. \
Copy the exact words from the message.
6. Extract ALL symptoms even if there are many. Do not stop after finding a few.
7. When unsure, INCLUDE with confidence="low" rather than omit. \
Missing a real symptom is clinically more dangerous than flagging an uncertain one.
8. Each symptom can appear at most once. Do not duplicate.
9. code and label_it MUST match — use the exact pairing from the catalog.
10. If the parent describes MULTIPLE distinct symptoms, extract ALL of them. \
For example, if a child has both a limp AND knee pain, extract BOTH. \
If a child snores AND has breathing pauses, extract BOTH russamento AND apnee.

## Italian negation markers
non, niente, nessun/a/o, senza, mai, nemmeno, neanche, neppure, mica, \
per niente, affatto

## Italian temporal resolution markers
è passato/a, è finito/a, è guarito/a, si è risolto/a, era...ma ora, \
ormai, non...più, settimana scorsa...passata, era...adesso no

## Symptom catalog
{catalog_block}
"""


def format_symptom_entry(s: EnrichedSymptom) -> str:
    """Format a single symptom for embedding in the system prompt."""
    lines = [
        f"---",
        f"Code: {s.code}",
        f"Label (IT): {s.label_it}",
        f"Label (EN): {s.label_en}",
        f"Triage: {s.triage_depth}",
        f"Definition: {s.short_definition}",
    ]
    if s.examples_it:
        examples = ", ".join(f'"{e}"' for e in s.examples_it[:8])
        lines.append(f"Examples: {examples}")
    if s.synonyms_it:
        lines.append(f"Synonyms: {', '.join(s.synonyms_it[:6])}")
    if s.negation_patterns_it:
        negs = ", ".join(f'"{n}"' for n in s.negation_patterns_it[:4])
        lines.append(f"Negation patterns: {negs}")
    if s.disambiguation:
        lines.append(f"Disambiguation: {s.disambiguation}")
    lines.append("---")
    return "\n".join(lines)


def build_baseline_prompt(catalog: list[EnrichedSymptom]) -> str:
    """Build the full baseline system prompt with all symptoms."""
    catalog_block = "\n\n".join(format_symptom_entry(s) for s in catalog)
    return BASELINE_SYSTEM_PROMPT.format(catalog_block=catalog_block)


# ---------------------------------------------------------------------------
# Stage 4: constrained extraction with top-K candidates
# ---------------------------------------------------------------------------

STAGE4_SYSTEM_PROMPT = """\
You are a pediatric symptom extractor analyzing messages from Italian parents.

## Task
From the parent's message, identify which of the candidate symptoms below are \
CURRENTLY PRESENT and ACTIVELY ASSERTED. For each identified symptom, provide \
the exact text span that evidences it.

## Rules
1. ONLY output symptoms from the candidate list below. You cannot invent labels.
2. EXCLUDE negated symptoms: "non ha la febbre" → fever is NOT present.
3. EXCLUDE past-resolved symptoms: "la febbre di ieri è passata" → fever is NOT present. \
Place it in excluded with reason "past_resolved".
4. INCLUDE hedged symptoms but mark them: "credo gli faccia male" → present, hedged=true, \
confidence="medium".
5. evidence_span MUST be a verbatim substring of the parent's message.
6. For confidence:
   - high: symptom explicitly stated with clear evidence
   - medium: symptom implied or described indirectly, or hedged
   - low: symptom ambiguously present, could be something else
7. When in doubt, INCLUDE the symptom with low confidence rather than omitting it.
8. code and label_it MUST match — use the exact pairing from the candidate list.

## Italian negation markers
non, niente, nessun/a/o, senza, mai, nemmeno, neanche, neppure, mica, \
per niente, affatto

## Italian temporal resolution markers
è passato/a, è finito/a, è guarito/a, si è risolto/a, era...ma ora, \
ormai, non...più, settimana scorsa...passata

## Candidate symptoms
{candidates_block}
"""


def build_stage4_prompt(candidates: list[EnrichedSymptom]) -> str:
    """Build Stage 4 system prompt with only the top-K candidates."""
    candidates_block = "\n\n".join(format_symptom_entry(s) for s in candidates)
    return STAGE4_SYSTEM_PROMPT.format(candidates_block=candidates_block)


# ---------------------------------------------------------------------------
# Stage 1: clause segmentation
# ---------------------------------------------------------------------------

SEGMENTATION_PROMPT = """\
Split the following Italian parent message into individual clauses. \
Each clause should contain one idea, symptom, or statement.

Rules:
- Preserve negation markers with their clause (e.g., "non ha la febbre" stays together)
- Preserve temporal markers with their clause (e.g., "da ieri sera ha la febbre")
- Do not reduce to keywords — keep full clauses intact
- Return a JSON array of strings

Message: {message}
"""


# ---------------------------------------------------------------------------
# Stage 2c: LLM free extraction
# ---------------------------------------------------------------------------

STAGE2C_EXTRACTION_PROMPT = """\
List every symptom, clinical sign, or physical complaint mentioned in this \
Italian parent message. Include:
- Explicitly stated symptoms
- Implied symptoms (e.g., "respira male" implies breathing difficulty)
- Negated symptoms (e.g., "non ha la febbre" — list "febbre" with a note it's negated)

Return a JSON array of objects, each with:
- "phrase": the short Italian phrase describing the symptom
- "negated": true if the symptom is negated in the message

Message: {message}
"""
