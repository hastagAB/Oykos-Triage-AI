"""Stage 1: Clause segmentation — rule-based with LLM fallback."""

from __future__ import annotations

import re

from ..models import Clause

NEGATION_MARKERS = {
    "non", "niente", "nessuno", "nessuna", "nessun",
    "senza", "mai", "nemmeno", "neanche", "neppure",
    "mica", "affatto",
}

TEMPORAL_MARKERS = [
    r"\bieri\b", r"\bstamattina\b", r"\bstasera\b", r"\bstanotte\b",
    r"\bda ieri\b", r"\bda stamattina\b", r"\bda qualche giorno\b",
    r"\bda oggi\b", r"\boggi\b", r"\bsettimana scorsa\b",
    r"\bil mese scorso\b", r"\bè passato\b", r"\bè passata\b",
    r"\bè finito\b", r"\bè finita\b", r"\bè guarito\b", r"\bè guarita\b",
    r"\bormai\b", r"\bnon\.\.\. *più\b", r"\bnon.*più\b",
]

CONJUNCTIONS = re.compile(
    r"(?:,\s*(?:e|ma|però|mentre|quando|perché|anche|poi|inoltre|oppure)\s)"
    r"|(?:\.\s+)"
    r"|(?:;\s*)"
    r"|(?:,\s+(?=[A-Z]))",
    re.IGNORECASE,
)


class RuleBasedSegmenter:
    """Segment Italian parent messages into clauses preserving negation and temporality."""

    def segment(self, message: str) -> list[Clause]:
        message = message.strip()
        if not message:
            return []

        if len(message) < 60 and "," not in message and "." not in message:
            return [self._make_clause(message)]

        parts = CONJUNCTIONS.split(message)
        parts = [p.strip() for p in parts if p and p.strip()]

        if not parts:
            return [self._make_clause(message)]

        clauses = []
        for part in parts:
            if len(part) < 3:
                if clauses:
                    prev = clauses[-1]
                    clauses[-1] = self._make_clause(prev.text + " " + part)
                continue
            clauses.append(self._make_clause(part))

        return clauses if clauses else [self._make_clause(message)]

    def _make_clause(self, text: str) -> Clause:
        text = text.strip().rstrip(".")
        low = text.lower()

        has_negation = any(
            re.search(rf"\b{neg}\b", low) for neg in NEGATION_MARKERS
        )

        temporal_marker = None
        for pattern in TEMPORAL_MARKERS:
            m = re.search(pattern, low)
            if m:
                temporal_marker = m.group()
                break

        return Clause(
            text=text,
            has_negation=has_negation,
            temporal_marker=temporal_marker,
        )
