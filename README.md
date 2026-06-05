# Oykos Triage AI

Pediatric symptom extraction from Italian parent messages using base LLMs. Replaces a fine-tuned model with a prompt-based system that achieves **97.6% accuracy** across 860 test cases (GPT-5.5), with zero hallucinated labels.

## How It Works

A parent writes a message in Italian describing their child's symptoms. The system extracts structured symptoms mapped to a fixed catalog of 80 canonical labels, handling negation, past-resolved symptoms, and multi-symptom messages.

```
Input:  "Da ieri sera ha la febbre alta e non vuole mangiare"
Output: [Febbre (high confidence), Inappetenza (high confidence)]
```

The system uses a single LLM call with the full symptom catalog in the system prompt. Enum-constrained structured output makes hallucinated labels mechanically impossible.

## Quick Start

```bash
# 1. Install
pip install -e ".[openai]"     # or: .[anthropic] / .[gemini] / .[all]

# 2. Set your API key
cp .env.example .env
# Edit .env with your key

# 3. Extract symptoms
python cli.py extract "Mio figlio ha la febbre e tossisce" --provider openai --model gpt-5.5-2026-04-23

# 4. Run evaluation
python cli.py evaluate --provider openai --model gpt-5.5-2026-04-23 --max-cases 50
```

## Supported Providers

| Provider | Models | Install |
|----------|--------|---------|
| OpenAI | gpt-5.5-2026-04-23, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano | `pip install -e ".[openai]"` |
| Anthropic | claude-sonnet-4-6, claude-sonnet-4-5, claude-opus-4-8, claude-opus-4-6, claude-opus-4-5 | `pip install -e ".[anthropic]"` |
| Google | gemini-2.5-flash, gemini-2.5-pro | `pip install -e ".[gemini]"` |

Switch providers with a flag — no code changes:

```bash
python cli.py extract "Febbre alta" --provider openai --model gpt-5.5-2026-04-23
python cli.py extract "Febbre alta" --provider anthropic --model claude-sonnet-4-6
python cli.py extract "Febbre alta" --provider gemini --model gemini-2.5-flash
```

## CLI Commands

```bash
# Extract symptoms from a message
python cli.py extract "message" --provider openai --model gpt-5.5-2026-04-23 [--verbose]

# Run evaluation on test dataset (860 cases)
python cli.py evaluate --provider openai --model gpt-5.5-2026-04-23 [--max-cases N] [--output results.json]

# Filter evaluation by case type
python cli.py evaluate --case-types positive,negation

# Enrich symptom catalog (generate synonyms, phrasings)
python cli.py enrich-catalog --provider openai

# Build embedding index (for pipeline mode)
python cli.py build-index
```

## Output Format

```json
{
  "symptoms": [
    {
      "code": "SI001",
      "label_it": "Febbre",
      "evidence_span": "ha la febbre alta",
      "confidence": "high",
      "negated": false,
      "hedged": false,
      "temporal_status": "current",
      "onset": "Da ieri sera"
    }
  ],
  "excluded": [
    {
      "code": "SI007",
      "label_it": "Tosse",
      "reason": "past_resolved",
      "evidence_span": "la tosse è passata"
    }
  ]
}
```

## Evaluation Results

Scored on 860 Italian parent messages. A message is **correct** only if every symptom is extracted exactly — nothing missed, nothing added.

| Model | Correct | Accuracy |
|-------|---------|----------|
| GPT-5.5 (OpenAI) | 839 / 860 | **97.6%** |
| Claude Sonnet 4.6 (Anthropic) | 826 / 860 | 96.0% |
| Claude Opus 4.8 (Anthropic) | 823 / 860 | 95.7% |
| Claude Opus 4.5 (Anthropic) | 821 / 860 | 95.5% |
| Claude Sonnet 4.5 (Anthropic) | 815 / 860 | 94.8% |
| Claude Opus 4.6 (Anthropic) | 805 / 860 | 93.6% |
| GPT-5.4 (OpenAI) | 803 / 860 | 93.4% |
| GPT-5.4 Mini (OpenAI) | 781 / 860 | 90.8% |
| GPT-5.4 Nano (OpenAI) | 717 / 860 | 83.4% |

Negation accuracy: **100%** — models correctly ignore negated symptoms ("non ha la febbre") every time.

See [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) for full results and per-model error analysis.

## Project Structure

```
symptom_extraction/         # Core package
  __init__.py               # Public API (OykosExtractor, models, providers)
  config.py                 # Configuration loader
  models.py                 # Pydantic models (shared types)
  llm/                      # Provider-agnostic LLM layer
    base.py                 #   Abstract interface
    openai_provider.py      #   OpenAI
    anthropic_provider.py   #   Anthropic
    gemini_provider.py      #   Google Gemini
  catalog/                  # Symptom catalog management
    loader.py               #   Load & validate catalog
    enrich.py               #   LLM-based enrichment
  pipeline/                 # Extraction pipeline
    orchestrator.py         #   Main entry point
    stage1_segmentation.py  #   Clause segmentation
    stage2a_dense.py        #   Dense retrieval
    stage2b_bm25.py         #   BM25 lexical retrieval
    stage2c_llm_extract.py  #   LLM extraction + NN
    stage3_fusion.py        #   Reciprocal Rank Fusion
    stage4_extraction.py    #   Constrained extraction
    stage5_gating.py        #   Confidence gating
  evaluation/               # Evaluation harness
    metrics.py              #   F1, precision, recall
    runner.py               #   Batch runner
    report.py               #   Console + JSON reports
  prompts/
    templates.py            #   All prompt templates

cli.py                      # CLI entry point
config.yaml                 # Default configuration
data/
  catalog/                  # Runtime symptom catalog (80 symptoms)
  eval/                     # Evaluation dataset (861 cases)
docs/                       # Architecture & evaluation docs
scripts/                    # Dev/data pipeline scripts
```

## Integration

```python
import asyncio
from symptom_extraction import OykosExtractor

# Auto-configures provider from config.yaml + .env
extractor = OykosExtractor.from_config()
result = asyncio.run(extractor.extract("Mio figlio ha la febbre"))

for symptom in result.confirmed:
    print(f"{symptom.code} {symptom.label_it}: {symptom.evidence_span}")
```

Or with explicit control:

```python
from symptom_extraction import load_config, load_catalog, OpenAIProvider, PipelineOrchestrator

config = load_config()
provider = OpenAIProvider(api_key="sk-...", default_model="gpt-5.5-2026-04-23")
catalog = load_catalog()
pipeline = PipelineOrchestrator(config, provider, catalog)

result = asyncio.run(pipeline.run("Mio figlio ha la febbre"))
```

## Documentation

- [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) — Full architecture, input/output specs, design decisions
- [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) — Detailed evaluation results and error analysis
- [TEST_DATASET.md](docs/TEST_DATASET.md) — Test dataset structure, sources, and methodology
