"""Abstract LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    calls: int = 0

    def merge(self, other: LLMUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.calls += other.calls


@dataclass
class LLMResponse:
    content: str = ""
    parsed: BaseModel | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)


class LLMProvider(ABC):
    """Provider-agnostic interface for LLM calls."""

    @abstractmethod
    async def extract_structured(
        self,
        system_prompt: str,
        user_message: str,
        response_schema: type[BaseModel],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        cache_system_prompt: bool = True,
    ) -> LLMResponse:
        """Call the LLM and parse the response into a Pydantic model.

        The provider must enforce the schema so the model cannot emit
        labels outside the allowed set (enum constraint).
        """
        ...

    @abstractmethod
    async def extract_text(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Call the LLM and return raw text."""
        ...
