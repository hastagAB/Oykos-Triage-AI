"""CLI entry point for the Oykos Triage AI symptom extraction pipeline."""

from __future__ import annotations

import asyncio
import json
import sys

import click

from symptom_extraction.config import get_api_key, load_config
from symptom_extraction.catalog.loader import load_catalog
from symptom_extraction.pipeline.orchestrator import PipelineOrchestrator


PROVIDERS = ["openai", "anthropic", "gemini"]


def _get_provider(config):
    """Create an LLM provider instance from config."""
    api_key = get_api_key(config.provider)

    if config.provider == "anthropic":
        from symptom_extraction.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, default_model=config.frontier_model)

    if config.provider == "openai":
        from symptom_extraction.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key, default_model=config.frontier_model)

    if config.provider == "gemini":
        from symptom_extraction.llm.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=api_key, default_model=config.frontier_model)

    raise ValueError(f"Unknown provider: {config.provider}")


def _apply_overrides(config, provider=None, model=None, mode=None, concurrency=None):
    """Apply CLI flag overrides to config."""
    if provider:
        config.provider = provider
    if model:
        config.frontier_model = model
    if mode:
        config.mode = mode
    if concurrency:
        config.max_concurrency = concurrency


@click.group()
def cli():
    """Oykos Triage AI — Pediatric Symptom Extraction Pipeline."""
    pass


@cli.command()
@click.argument("message")
@click.option("--provider", type=click.Choice(PROVIDERS), default=None, help="LLM provider")
@click.option("--model", default=None, help="Model name (e.g., gpt-4o, claude-sonnet-4-20250514, gemini-2.5-flash)")
@click.option("--mode", type=click.Choice(["baseline", "pipeline"]), default=None)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--verbose", is_flag=True, help="Include metadata in output")
def extract(message, provider, model, mode, config_path, verbose):
    """Extract symptoms from a single Italian parent message."""
    config = load_config(config_path)
    _apply_overrides(config, provider=provider, model=model, mode=mode)

    llm = _get_provider(config)
    catalog = load_catalog()
    pipeline = PipelineOrchestrator(config, llm, catalog)

    result = asyncio.run(pipeline.run(message))

    output = {
        "symptoms": [s.model_dump() for s in result.confirmed],
        "flagged_for_review": [s.model_dump() for s in result.flagged_for_review],
        "excluded": [e.model_dump() for e in result.excluded],
    }
    if verbose:
        output["metadata"] = result.pipeline_metadata.model_dump()

    click.echo(json.dumps(output, ensure_ascii=False, indent=2))


@cli.command()
@click.option("--dataset", type=click.Path(exists=True), default="data/eval/test_dataset.jsonl")
@click.option("--provider", type=click.Choice(PROVIDERS), default=None, help="LLM provider")
@click.option("--model", default=None, help="Model name (e.g., gpt-4o, claude-sonnet-4-20250514, gemini-2.5-flash)")
@click.option("--mode", type=click.Choice(["baseline", "pipeline"]), default=None)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--max-cases", type=int, default=None, help="Limit number of cases to evaluate")
@click.option("--case-types", type=str, default=None, help="Filter: positive,negation,past_resolved,multi_symptom")
@click.option("--output", "output_path", type=click.Path(), default=None, help="Save JSON report to file")
@click.option("--concurrency", type=int, default=None, help="Max parallel API calls")
def evaluate(dataset, provider, model, mode, config_path, max_cases, case_types, output_path, concurrency):
    """Run batch evaluation on the test dataset."""
    from symptom_extraction.evaluation.runner import EvaluationRunner

    config = load_config(config_path)
    _apply_overrides(config, provider=provider, model=model, mode=mode, concurrency=concurrency)

    llm = _get_provider(config)
    catalog = load_catalog()
    pipeline = PipelineOrchestrator(config, llm, catalog)
    runner = EvaluationRunner(pipeline, config, catalog)

    type_filter = case_types.split(",") if case_types else None

    report = asyncio.run(runner.run_dataset(
        dataset_path=dataset,
        max_cases=max_cases,
        case_types=type_filter,
    ))

    from symptom_extraction.evaluation.report import print_report, save_json_report
    print_report(report)

    if output_path:
        save_json_report(report, output_path)
        click.echo(f"\nJSON report saved to: {output_path}")


@cli.command("enrich-catalog")
@click.option("--provider", type=click.Choice(PROVIDERS), default=None)
@click.option("--model", default=None)
@click.option("--config", "config_path", type=click.Path(), default=None)
def enrich_catalog_cmd(provider, model, config_path):
    """Bootstrap enriched catalog with LLM-generated synonyms and phrasings."""
    from symptom_extraction.catalog.enrich import CatalogEnricher

    config = load_config(config_path)
    _apply_overrides(config, provider=provider, model=model)

    llm = _get_provider(config)
    catalog = load_catalog()
    enricher = CatalogEnricher(llm, config)
    asyncio.run(enricher.enrich_all(catalog))
    click.echo("Enriched catalog saved.")


@cli.command("build-index")
@click.option("--config", "config_path", type=click.Path(), default=None)
def build_index_cmd(config_path):
    """Pre-compute symptom embeddings for pipeline mode."""
    from symptom_extraction.embeddings.index import SymptomVectorIndex

    config = load_config(config_path)
    catalog = load_catalog()
    index = SymptomVectorIndex(catalog, config)
    index.build_and_save()
    click.echo("Embedding index built and saved.")


if __name__ == "__main__":
    cli()
