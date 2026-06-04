"""Configuration loader — YAML file with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .models import PipelineConfig

load_dotenv()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

PROVIDER_DEFAULTS = {
    "anthropic": {"frontier": "claude-sonnet-4-20250514", "cheap": "claude-3-5-haiku-20241022"},
    "openai": {"frontier": "gpt-4o", "cheap": "gpt-4o-mini"},
    "gemini": {"frontier": "gemini-2.5-flash", "cheap": "gemini-2.0-flash"},
}

API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    provider = os.environ.get("OYKOMED_PROVIDER", raw.get("provider", "openai"))
    mode = os.environ.get("OYKOMED_MODE", raw.get("mode", "baseline"))

    provider_cfg = raw.get(provider, {})
    pipeline_cfg = raw.get("pipeline", {})
    gating_cfg = raw.get("gating", {})
    eval_cfg = raw.get("evaluation", {})
    embedding_cfg = raw.get("embedding", {})

    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])

    return PipelineConfig(
        mode=mode,
        provider=provider,
        frontier_model=provider_cfg.get("frontier_model", defaults["frontier"]),
        cheap_model=provider_cfg.get("cheap_model", defaults["cheap"]),
        embedding_model=embedding_cfg.get("model", "intfloat/multilingual-e5-large-instruct"),
        top_k_candidates=pipeline_cfg.get("top_k_candidates", 15),
        rrf_k_constant=pipeline_cfg.get("rrf_k_constant", 60),
        high_confidence_threshold=gating_cfg.get("high_threshold", 0.7),
        review_confidence_threshold=gating_cfg.get("review_threshold", 0.4),
        temperature=float(raw.get("temperature", 0.0)),
        cache_system_prompt=raw.get("cache_system_prompt", True),
        use_few_shot=raw.get("use_few_shot", False),
        use_verifier=raw.get("use_verifier", False),
        recall_floor=eval_cfg.get("recall_floor", 0.85),
        min_support_for_floor=eval_cfg.get("min_support_for_floor", 8),
        max_concurrency=int(raw.get("max_concurrency", 5)),
    )


def get_api_key(provider: str) -> str:
    env_var = API_KEY_ENV.get(provider, f"{provider.upper()}_API_KEY")
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(f"API key not found. Set {env_var} in your environment or .env file.")
    return key
