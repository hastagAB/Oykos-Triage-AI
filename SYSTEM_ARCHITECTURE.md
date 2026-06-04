# System Architecture: Pediatric Symptom Extraction MVP

## 1. Problem Statement

The fine-tuned model previously used for pediatric symptom identification (triage) has been removed. This system replaces it with a **base-model-only pipeline** that extracts structured symptoms from Italian parent messages, mapping them to a fixed catalog of 80 canonical symptom labels.

The system must handle colloquial Italian text (WhatsApp-style), detect negation and temporal resolution, and produce enum-constrained JSON output with zero hallucinated labels.

---

## 2. Architecture Overview

The MVP uses a **single-call baseline architecture**: one LLM call with the full 80-symptom catalog embedded in a cached system prompt, combined with structured output (JSON schema with enum constraints) to mechanically prevent hallucinated labels.

```
Parent Message (Italian)
        |
        v
+------------------+
|  System Prompt    |  <-- Cached (~27K chars)
|  - Instructions   |      Contains all 80 symptoms
|  - Negation rules |      with definitions
|  - 80 symptoms    |
|  - Enum schema    |
+------------------+
        |
        v
+------------------+
|   LLM Call       |  GPT-5.5 (or any frontier model)
|   (structured    |  Provider-agnostic abstraction
|    output)       |  Enum-constrained JSON response
+------------------+
        |
        v
+------------------+
|  Post-processing |  Filter negated/past-resolved
|  & Validation    |  Validate Pydantic model
+------------------+
        |
        v
  Structured JSON Output
```

A full 5-stage pipeline (clause segmentation, parallel retrieval, RRF fusion, constrained extraction, confidence gating) is also implemented but not required for the baseline which already achieves 95.6% F1.

---

## 3. Input Specification

### 3.1 Input Format

A single Italian-language message from a parent describing their child's symptoms.

**Characteristics:**
- Language: Italian (colloquial, often informal)
- Length: 5-200 words typically
- Style: WhatsApp/email to pediatrician
- Content: May contain multiple symptoms, negated symptoms, past-resolved symptoms, hedged statements, misspellings, dialect
- Names: Often includes child's name and age

**Examples:**
```
"Vittoria ha la febbre"

"Da ieri sera ha la febbre alta, non vuole mangiare, e credo gli faccia 
male la pancia. La tosse di settimana scorsa per fortuna e passata."

"Teresa ha un po' di raffreddore, respiro gracchiante, tosse da fastidio"
```

### 3.2 Canonical Symptom Catalog

80 symptoms (SI001-SI080), each with:
- `code`: Unique identifier (SI001-SI080)
- `label_it`: Italian canonical label
- `label_en`: English translation
- `triage_depth`: Alta (high) or Media (medium)
- `short_definition`: Clinical definition in Italian

Source file: `data/test/symptom_catalog.json`

---

## 4. Output Specification

### 4.1 Output Schema

```json
{
  "symptoms": [
    {
      "code": "SI001",
      "label_it": "Febbre",
      "evidence_span": "ha la febbre alta",
      "negated": false,
      "hedged": false,
      "temporal_status": "current",
      "confidence": "high",
      "onset": "Da ieri sera"
    }
  ],
  "excluded": [
    {
      "code": "SI007",
      "label_it": "Tosse",
      "reason": "past_resolved",
      "evidence_span": "La tosse di settimana scorsa per fortuna e passata"
    }
  ]
}
```

### 4.2 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `code` | string | Canonical symptom code (SI001-SI080). Enum-constrained. |
| `label_it` | string | Italian label from catalog. Must match the code. |
| `evidence_span` | string | Verbatim substring from the parent's message. |
| `negated` | boolean | True if the symptom is explicitly denied. |
| `hedged` | boolean | True if the parent expresses uncertainty ("credo", "forse"). |
| `temporal_status` | enum | `current`, `past_resolved`, or `chronic`. |
| `confidence` | enum | `high` (explicit), `medium` (indirect/hedged), `low` (ambiguous). |
| `onset` | string | Temporal onset if mentioned (e.g., "da ieri sera"). |
| `reason` | string | Why a symptom was excluded: `negated`, `past_resolved`, `below_threshold`. |

### 4.3 Enum Constraint Mechanism

The output schema dynamically generates a Pydantic model where the `code` and `label_it` fields are `Literal` types constrained to only the allowed symptom codes/labels. This is enforced at the API level:

- **OpenAI**: `response_format` with strict JSON schema mode
- **Anthropic**: `tool_use` with `tool_choice` forced to a single tool whose `input_schema` is the constrained model

This makes hallucinated labels **mechanically impossible** — the model cannot emit a token outside the allowed set.

---

## 5. System Components

### 5.1 Project Structure

```
symptom_extraction/
  __init__.py
  config.py                 # YAML config + env variable loading
  models.py                 # All Pydantic models (shared vocabulary)
  
  llm/
    base.py                 # Abstract LLMProvider interface
    anthropic_provider.py   # Claude implementation (tool_use + prompt caching)
    openai_provider.py      # OpenAI implementation (response_format)
  
  catalog/
    loader.py               # Load symptom_catalog.json -> EnrichedSymptom
    enrich.py               # LLM-based catalog enrichment (synonyms, phrasings)
  
  pipeline/
    orchestrator.py         # Runs baseline or full pipeline
    stage1_segmentation.py  # Rule-based clause segmentation
    stage2a_dense.py        # Dense embedding retrieval
    stage2b_bm25.py         # BM25 lexical retrieval
    stage2c_llm_extract.py  # LLM extraction + nearest-neighbor
    stage3_fusion.py        # Reciprocal Rank Fusion
    stage4_extraction.py    # Constrained final extraction
    stage5_gating.py        # Confidence gating & abstention
    verifier.py             # Optional verification pass
  
  embeddings/
    encoder.py              # SentenceTransformer wrapper
    index.py                # Pre-computed symptom vector index
  
  evaluation/
    metrics.py              # F1, precision, recall computation
    runner.py               # Batch evaluation over test dataset
    report.py               # Console + JSON reporting
  
  prompts/
    templates.py            # All prompt templates

cli.py                      # Click CLI entry point
config.yaml                 # Default configuration
```

### 5.2 LLM Abstraction Layer

The system is provider-agnostic. An abstract `LLMProvider` interface defines two methods:

- `extract_structured(system_prompt, user_message, response_schema)` — Returns a validated Pydantic model
- `extract_text(system_prompt, user_message)` — Returns raw text

Each provider implements structured output differently:

| Provider | Structured Output Mechanism | Prompt Caching |
|----------|---------------------------|----------------|
| Anthropic Claude | `tool_use` with forced `tool_choice` | `cache_control: {"type": "ephemeral"}` on system prompt |
| OpenAI GPT | `response_format` with strict JSON schema | Automatic for identical prefixes |

Both include retry logic (3 attempts, exponential backoff) via `tenacity`.

### 5.3 Prompt Architecture

The system prompt (~27,000 characters) contains:

1. **Role and task description** — "You are a pediatric symptom extractor"
2. **10 critical extraction rules** — Covering negation, past-resolved, hedging, evidence spans, completeness
3. **Italian negation markers** — non, niente, nessun/a/o, senza, mai, nemmeno, neanche, neppure, mica
4. **Italian temporal resolution markers** — e passato/a, e finito/a, ormai, non...piu
5. **Full 80-symptom catalog** — Each symptom with code, Italian label, English label, triage depth, clinical definition

The catalog is static and cached after the first call, so subsequent messages only pay for the user message tokens (~50-200 tokens per parent message vs ~9,500 tokens for the cached catalog).

### 5.4 Post-Processing

After the LLM returns the structured response:

1. **Negation filter**: Symptoms marked `negated=true` or `temporal_status=past_resolved` are moved from `symptoms` to `excluded`
2. **Pydantic validation**: The response is validated against the constrained schema
3. **Optional verification pass**: A second LLM call can review multi-symptom extractions to remove consequence symptoms (currently disabled as it hurts recall)

---

## 6. Full Pipeline Architecture (Implemented, Not Required for Baseline)

For cases where the baseline falls short, a 5-stage pipeline is available:

```
Parent message (Italian)
       |
       v
Stage 1: Clause Segmentation (rule-based)
       |   Splits on Italian clause boundaries
       |   Preserves negation and temporal markers
       v
Stage 2: Parallel Retrieval (3 methods per clause)
       |   2a: Dense embeddings (multilingual-e5-large-instruct)
       |   2b: BM25 lexical (Italian lemmatization via simplemma)
       |   2c: LLM extract + nearest-neighbor lookup
       v
Stage 3: RRF Fusion (pure code, no LLM)
       |   score(s) = sum(1 / (60 + rank_r(s)))
       |   Take top 15 candidates
       v
Stage 4: Constrained Extraction (frontier LLM)
       |   Same as baseline but with only top-K candidates
       |   Smaller enum = higher precision
       v
Stage 5: Confidence Gating (rule-based)
       |   Combines LLM confidence + retriever agreement
       |   confirmed / flagged_for_review / abstained
       v
Structured JSON Output
```

---

## 7. Configuration

All configuration is in `config.yaml` with environment variable overrides:

```yaml
provider: openai              # or "anthropic"
mode: baseline                # or "pipeline"

openai:
  frontier_model: gpt-5.5-2026-04-23
  cheap_model: gpt-4o-mini

anthropic:
  frontier_model: claude-sonnet-4-20250514
  cheap_model: claude-3-5-haiku-20241022

pipeline:
  top_k_candidates: 15
  rrf_k_constant: 60

gating:
  high_threshold: 0.7
  review_threshold: 0.4

evaluation:
  recall_floor: 0.85
  min_support_for_floor: 8
```

---

## 8. CLI Interface

```bash
# Extract symptoms from a single message
python cli.py extract "Mio figlio ha la febbre e tossisce" --mode baseline --provider openai --verbose

# Run batch evaluation on the test dataset
python cli.py evaluate --mode baseline --provider openai --output results.json --concurrency 10

# Enrich the symptom catalog with LLM-generated synonyms
python cli.py enrich-catalog --provider openai

# Pre-compute embedding vectors for pipeline mode
python cli.py build-index
```

---

## 9. Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API client |
| `openai` | OpenAI API client |
| `pydantic` | Data models, JSON schema generation, enum constraints |
| `click` | CLI framework |
| `pyyaml` | Configuration loading |
| `sentence-transformers` | Multilingual embeddings (pipeline mode) |
| `rank-bm25` | BM25 lexical retrieval (pipeline mode) |
| `simplemma` | Italian lemmatization (pipeline mode) |
| `tenacity` | Retry logic for API calls |
| `tqdm` | Progress bars for batch evaluation |
| `python-dotenv` | API key management |

---

## 10. Cost Model

### Baseline mode (per message):
- **Input tokens**: ~9,500 (catalog, cached after first call) + ~50-200 (parent message)
- **Output tokens**: ~100-400 (structured JSON response)
- **After caching**: Only ~50-200 fresh input tokens per message
- **Latency**: 1-4 seconds per message

### Pipeline mode (per message):
- **LLM calls**: 2 (Stage 2c cheap extraction + Stage 4 main extraction)
- **Compute**: Embedding similarity + BM25 (sub-second, local)
- **Latency**: 3-8 seconds per message

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single-call baseline over pipeline | 80 symptoms fit in context (~15K tokens). Frontier models handle this without lost-in-the-middle degradation. |
| Enum-constrained structured output | Mechanically prevents hallucinated labels. No post-hoc validation needed. |
| Provider-agnostic abstraction | Enables model comparison and future-proofing. Swap providers via config. |
| Recall-biased extraction rules | "When unsure, INCLUDE with low confidence" — missing a real symptom is clinically more dangerous than flagging an uncertain one. |
| Evidence span requirement | Forces grounding: every extracted symptom must cite the exact text that supports it. |
| Negation as separate output | Excluded symptoms with reasons are returned alongside extracted symptoms, providing full transparency. |
