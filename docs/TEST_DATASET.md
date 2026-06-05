# Test Dataset: Pediatric Symptom Extraction

## Overview

| Property | Value |
|----------|-------|
| **Total cases** | 861 |
| **Symptoms covered** | 80 / 80 (100%) |
| **Min test cases per symptom** | 8 |
| **Max test cases per symptom** | 20 |
| **Average test cases per symptom** | 10.5 |
| **Multi-symptom cases** | 90 |
| **Negation cases** | 80 |
| **Past-resolved cases** | 40 |
| **Average message length** | 81 characters |
| **Language** | Italian (colloquial, WhatsApp-style) |
| **File** | `data/eval/test_dataset.jsonl` |

---

## Sources

| Source | Cases | Description |
|--------|-------|-------------|
| Real user messages (`xlsx`) | 20 | Actual parent messages from production WhatsApp conversations with a pediatrician. Gold standard for realistic input. |
| Expert test prompts (`docx`) | 376 | Clinician-authored test prompts across 8 tables covering single symptoms, multi-symptom combinations, stress tests, and hard-mode non-obvious routing. |
| Synthetic hand-authored | 465 | 4 positive cases per symptom (320), 1 negation per symptom (80), 0.5 past-resolved per symptom (40), and 25 complex multi-symptom cases. Written to cover colloquial Italian variations. |

---

## Case Types

### Positive (320 cases)
Each of the 80 symptoms has 4 hand-authored positive cases. Each message clearly describes the symptom using different Italian phrasings, varying child ages, and different levels of formality.

**Example:**
```json
{
  "id": "syn_positive_0001",
  "message": "Da stamattina Marco scotta tantissimo, il termometro segna 39.2 e ha le guance rosse fuoco.",
  "expected_symptoms_canonical": ["Febbre"],
  "case_type": "positive"
}
```

### Negation (80 cases)
One case per symptom. The message explicitly denies the target symptom while describing other symptoms. The gold label set is empty — the system must NOT extract the negated symptom. Other symptoms present in the message are allowed.

Each case includes a `negated_symptom` field identifying which symptom must be absent from the output.

**Example:**
```json
{
  "id": "syn_negation_0001",
  "message": "Da ieri ha mal di testa e si sente debole ma niente febbre, l'ho misurato tre volte e la temperatura e normale.",
  "expected_symptoms_canonical": [],
  "case_type": "negation",
  "negated_symptom": "Febbre"
}
```

### Past-Resolved (40 cases)
The message describes a symptom that has already resolved. The system must NOT extract it. Each case includes a `resolved_symptom` field.

**Example:**
```json
{
  "id": "syn_past_resolved_0001",
  "message": "La febbre la settimana scorsa e passata da giorni, ora chiamo solo per il certificato.",
  "expected_symptoms_canonical": [],
  "case_type": "past_resolved",
  "resolved_symptom": "Febbre"
}
```

### Multi-Symptom (25 synthetic + 65 from expert prompts)
Messages containing 2-3 symptoms that must all be extracted. These test the system's ability to find multiple symptoms in complex, colloquial messages.

**Example:**
```json
{
  "id": "syn_multi_symptom_0001",
  "message": "Da ieri sera ha la febbre alta, non vuole mangiare e credo gli faccia male la pancia.",
  "expected_symptoms_canonical": ["Febbre", "Inappetenza", "Dolore addominale"],
  "case_type": "multi_symptom"
}
```

### Expert Test Sections (from docx)

| Section | Cases | Description |
|---------|-------|-------------|
| `standard_single_symptom_a` | 80 | One prompt per symptom, straightforward descriptions |
| `standard_single_symptom_b` | 60 | Additional single-symptom prompts with variations |
| `multi_symptom_sleep` | 20 | Multi-symptom combinations involving sleep disorders |
| `multi_symptom_metabolic` | 16 | Multi-symptom combinations involving metabolic symptoms |
| `multi_symptom_mixed` | 20 | Mixed multi-symptom combinations |
| `hard_mode_non_obvious_routing` | 20 | Symptoms described indirectly or with non-obvious phrasing |
| `stress_vertigini_capogiro` | 80 | Stress test: messages that could be confused with dizziness |
| `stress_prurito_cutaneo` | 80 | Stress test: messages that could be confused with skin itching |

---

## Record Schema

Each record in `test_dataset.jsonl` has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique case identifier (e.g., `up_001`, `doc_t0_r002`, `syn_positive_0001`) |
| `source` | string | Origin: `xlsx:UP enquiries`, `docx:table_N`, or `synthetic:author` |
| `section` | string | Test section (see table above) |
| `message` | string | Italian parent message (the input to the system) |
| `expected_symptoms_canonical` | string[] | Gold standard symptom labels (Italian canonical names) |
| `expected_symptoms_raw` | string | Original gold label text before canonicalization |
| `expected_symptoms_unresolved` | string[] | Labels that couldn't be mapped to the catalog |
| `all_labels_in_catalog` | boolean | True if all expected labels exist in the symptom catalog |
| `case_type` | string | `positive`, `negation`, `past_resolved`, `multi_symptom`, or empty |
| `negated_symptom` | string | (negation cases only) The symptom that must NOT be extracted |
| `resolved_symptom` | string | (past_resolved cases only) The symptom that must NOT be extracted |
| `verdict_original_run` | string | Result from the original fine-tuned model (historical) |
| `verdict_retrained_run` | string | Result from the retrained model (historical) |
| `notes` | string | Annotator notes |

---

## Gold Label Corrections

49 gold labels were corrected after the initial evaluation revealed under-annotation. The principle: if a message genuinely describes a symptom, it belongs in the gold set regardless of which symptom the test case was "designed" to test.

Corrections are documented in `scripts/fix_gold_labels.py` with per-case reasoning. The original dataset is preserved as `data/eval/test_dataset.jsonl.bak`.

**Common correction types:**
- Messages describing co-occurring symptoms (e.g., "russa e il respiro si ferma" — both Russamento and Apnee are present)
- Messages where the gold label didn't match the message content (e.g., gold said "Dolore muscolare" but message only described limping)
- Resolved symptoms marked as current in gold (e.g., "non ha piu febbre" — fever is resolved)

---

## Symptom Catalog

The canonical symptom list contains 80 symptoms (SI001-SI080), stored in `data/catalog/symptom_catalog.json`.

Each symptom has:
- **code**: SI001-SI080
- **label_it**: Italian canonical label (e.g., "Febbre")
- **label_en**: English translation (e.g., "Fever")
- **triage_depth**: "Alta" (high priority) or "Media" (medium priority)
- **short_definition**: Clinical definition in Italian for parents

High-priority (Alta) symptoms: 36 — these typically require immediate pediatric attention.
Medium-priority (Media) symptoms: 44 — these may require routine consultation.

---

## How to Run Evaluation

```bash
# Full evaluation (all 861 cases)
python cli.py evaluate --dataset data/eval/test_dataset.jsonl --provider openai --model gpt-5.5-2026-04-23

# Subset by case type
python cli.py evaluate --case-types positive,negation

# Limited cases for quick testing
python cli.py evaluate --max-cases 50

# Save JSON report
python cli.py evaluate --output results/my_evaluation.json

# Compare providers
python cli.py evaluate --provider anthropic --model claude-sonnet-4-6
python cli.py evaluate --provider anthropic --model claude-opus-4-8
python cli.py evaluate --provider gemini --model gemini-2.5-flash
```
