"""Oykos Triage AI — Pediatric symptom extraction from Italian parent messages."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import load_config, get_api_key
from .catalog.loader import load_catalog
from .models import (
    PipelineConfig,
    ExtractedSymptom,
    ExcludedSymptom,
    ExtractionResult,
    GatedResult,
    PipelineMetadata,
    EnrichedSymptom,
)
from .pipeline.orchestrator import PipelineOrchestrator
from .llm.base import LLMProvider, LLMResponse, LLMUsage


def __getattr__(name: str):
    if name == "OpenAIProvider":
        from .llm.openai_provider import OpenAIProvider
        return OpenAIProvider
    if name == "AnthropicProvider":
        from .llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider
    if name == "GeminiProvider":
        from .llm.gemini_provider import GeminiProvider
        return GeminiProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class OykosExtractor:
    """High-level facade for symptom extraction.

    Usage::

        from symptom_extraction import OykosExtractor

        extractor = OykosExtractor.from_config()
        result = await extractor.extract("Mio figlio ha la febbre")
        for s in result.confirmed:
            print(s.label_it, s.confidence)
    """

    def __init__(
        self,
        config: PipelineConfig,
        provider: LLMProvider,
        catalog: list[EnrichedSymptom] | None = None,
    ):
        if catalog is None:
            catalog = load_catalog()
        self._pipeline = PipelineOrchestrator(config, provider, catalog)

    @classmethod
    def from_config(
        cls,
        config_path: str | None = None,
        api_key: str | None = None,
    ) -> OykosExtractor:
        """Create an extractor from config file, auto-selecting the provider."""
        config = load_config(config_path)
        key = api_key or get_api_key(config.provider)

        if config.provider == "anthropic":
            from .llm.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(api_key=key, default_model=config.frontier_model)
        elif config.provider == "gemini":
            from .llm.gemini_provider import GeminiProvider
            provider = GeminiProvider(api_key=key, default_model=config.frontier_model)
        else:
            from .llm.openai_provider import OpenAIProvider
            provider = OpenAIProvider(api_key=key, default_model=config.frontier_model)

        return cls(config, provider)

    async def extract(self, message: str) -> GatedResult:
        """Extract symptoms from an Italian parent message."""
        return await self._pipeline.run(message)


__all__ = [
    "__version__",
    "load_config",
    "get_api_key",
    "load_catalog",
    "PipelineConfig",
    "ExtractedSymptom",
    "ExcludedSymptom",
    "ExtractionResult",
    "GatedResult",
    "PipelineMetadata",
    "EnrichedSymptom",
    "PipelineOrchestrator",
    "LLMProvider",
    "LLMResponse",
    "LLMUsage",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OykosExtractor",
]
