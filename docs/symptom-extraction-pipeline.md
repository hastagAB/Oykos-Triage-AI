# Pediatric Symptom Extraction Without Fine-Tuning

A base-model-only system that preserves fine-tuned-level accuracy for multi-symptom extraction from Italian parent messages, under strict label constraints and limited context.

## 1. Problem in one line

Replace the removed fine-tuned triage model with base models, keeping the same recall and precision across the full 80+ symptom label set, with valid JSON output and clinical-grade reliability.

## 2. Design principle

Two facts shape everything below.

The accuracy ceiling is set by retrieval recall. If a symptom is not surfaced as a candidate before the final extraction step, no downstream model can recover it.

The precision ceiling is set by catalog quality and the final-extraction prompt, not by adding more pipeline stages. Most of the real work is building the symptom catalog, not wiring components.

Start simple. Add complexity only where the evaluation proves a specific failure, and only for the part that fails.

## 3. Two designs

### 3.1 The simple baseline (build this first)

A single LLM call with the full catalog in a cached system prompt and an enum-constrained JSON schema.

```
System prompt: instructions + full catalog (80+ symptoms with definitions,
                examples, negation patterns, disambiguation notes)
User message:  the original parent message
Output schema: JSON, label field is an enum of all canonical labels
```

Why this is often enough with current models:

- 80 symptoms with definitions and examples is roughly 8K to 15K tokens. Frontier models handle this length without the lost-in-the-middle degradation seen in older models.
- Native structured output with an enum makes hallucinated labels mechanically impossible. The model cannot emit a token outside the allowed set.
- An explicit negation instruction plus a required `evidence_span` per label forces grounding, which suppresses false positives.
- Prompt caching on the static catalog makes the per-message cost low after the first call.

Ship this, evaluate it on the held-out test set, and measure per-message accuracy. Only move to the full pipeline for the symptoms that are causing misses.

### 3.2 The full pipeline (add only where the baseline fails)

Five stages: segment, retrieve in parallel, fuse, extract under constraint, gate.

```
Parent message (Italian)
   |
   v
Stage 1  Clause segmentation
   |
   v
Stage 2  Parallel retrieval per clause
           2a Dense (multilingual embeddings)
           2b BM25 (Italian synonym lexicon)
           2c LLM extract then nearest-neighbor
   |
   v
Stage 3  RRF fusion -> top K candidates (15 to 20)
   |
   v
Stage 4  Constrained final extraction (frontier base LLM)
   |
   v
Stage 5  Abstention and confidence gating
   |
   v
Validated JSON + clinician review queue
```

## 4. Stage by stage

### Stage 1: Clause segmentation

Purpose: long colloquial messages have low signal-to-noise. Segmenting raises retrieval recall on multi-symptom messages.

The key rule: do not reduce to keywords. Caveman keyword reduction destroys co-occurrence and negation. "Si sveglia piangendo e si tocca l'orecchio" is an ear-pain signal as a whole clause; the tokens alone are not. "Non ha piu la febbre" becomes a false positive if you keep only "febbre".

Segment by clause, keeping each clause intact with its negation and temporal markers.

- Tool: cheap LLM (for example Haiku) or deterministic rules.
- In: the full message.
- Out: a list of clauses, each preserving negation and temporality.

Example:

```
In:  "Da ieri sera ha la febbre alta, non vuole mangiare, e credo gli faccia
      male la pancia. La tosse di settimana scorsa per fortuna e passata."

Out: [
  "Da ieri sera ha la febbre alta",
  "non vuole mangiare",
  "credo gli faccia male la pancia",
  "La tosse di settimana scorsa per fortuna e passata"   # past, resolved
]
```

### Stage 2: Parallel retrieval per clause

Purpose: maximize recall. This is the stage that sets the accuracy ceiling. Run three retrievers on every clause because each catches a different failure class.

#### 2a. Dense retrieval

- Tool: multilingual embedding model with strong Italian support (BGE-M3 or multilingual-e5-large).
- In: clause text, compared against precomputed catalog vectors. Each symptom is embedded as a single vector built from its label, definition, example phrasings, and synonyms.
- Out: top 10 labels ranked by cosine similarity.
- Catches: paraphrases and semantic matches.

#### 2b. BM25 lexical

- Tool: BM25 over a curated Italian synonym lexicon, one list of surface forms per symptom (for example febbre, temperatura alta, scotta, bollente).
- In: clause tokens.
- Out: top 10 labels ranked by BM25 score.
- Catches: exact and morphological surface forms that dense retrieval misses.

#### 2c. LLM extract then nearest-neighbor

- Tool: small LLM for free extraction, then an embedding nearest-neighbor lookup.
- In: clause text. Prompt: list every symptom or clinical sign mentioned, including implied or negated ones, as short Italian phrases.
- Out: extracted phrases, each mapped to its top 5 nearest symptom labels.
- Catches: implied symptoms with no matching surface form, for example "respira male e fa rumore quando dorme" mapping toward dyspnea or stridor.

### Stage 3: RRF fusion

Purpose: merge the ranked lists into one, preserving cross-retriever rank signal. Naive union throws away ranking; a label that ranked first in two retrievers should outrank one that appeared once at rank 15.

- Tool: Reciprocal Rank Fusion, pure code, no LLM.
- In: ranked lists from 2a, 2b, 2c, per clause, then merged across clauses.
- Out: a deduplicated ranked list of candidate labels. Take the top K, where K is 15 to 20.

Formula:

```
For each candidate symptom s:
  score(s) = sum over retrievers r of  1 / (60 + rank_r(s))
Sort descending, take top K.
```

Tune K empirically until recall plateaus on the held-out set. K too small misses real symptoms ranked just outside the cut. K too large widens the candidate list and raises confident false positives in Stage 4.

### Stage 4: Constrained final extraction

Purpose: decide which candidates are actually asserted, handle negation and temporality, and emit valid constrained JSON. This is the only stage that sees the original full message, because segmentation was for retrieval and final judgment needs full context.

- Tool: frontier base LLM (for example Opus 4.7 or GPT-5) with structured output.
- In, three parts:
  - the original untouched message,
  - the top K candidates with full definitions, examples, and negation patterns,
  - a JSON schema whose label field is an enum restricted to those K candidates.
- Constraint: the model cannot emit a label outside the enum, so hallucinated labels are impossible.
- Instruction: exclude negated symptoms, exclude past-resolved symptoms, require an evidence_span per extracted symptom.
- Out: structured JSON with evidence_span, negated, and confidence per symptom.

Example output for the running message:

```json
{
  "symptoms": [
    {"label": "fever", "evidence_span": "ha la febbre alta", "onset": "ieri sera", "negated": false, "confidence": "high"},
    {"label": "appetite_loss", "evidence_span": "non vuole mangiare", "negated": false, "confidence": "high"},
    {"label": "abdominal_pain", "evidence_span": "gli faccia male la pancia", "negated": false, "hedged": true, "confidence": "medium"}
  ],
  "excluded": [
    {"label": "cough", "reason": "past_resolved", "evidence_span": "tosse di settimana scorsa e passata"}
  ]
}
```

If this stage is done well (original message, constrained schema, explicit negation and temporality instructions, required evidence span), it also does the verification job. Do not add a separate verifier stage unless evaluation shows precision dropping below threshold. That is the over-engineering to avoid.

### Stage 5: Abstention and confidence gating

Purpose: a clinical system must abstain rather than guess. Low-confidence symptoms go to a clinician, not silently dropped or silently kept.

- Tool: rule-based, no LLM.
- In: the structured JSON from Stage 4 plus retrieval scores from Stage 3. Agreement across retrievers raises confidence (a label surfaced by all three is stronger than one surfaced by one).
- Out: a confirmed list (high confidence) and a flagged-for-review list.

## 5. The symptom catalog (the real moat)

A single source of truth that feeds four of the five stages. Spend disproportionate effort here.

Per symptom:

- canonical label,
- Italian definition,
- 5 to 10 real Italian parent phrasings,
- synonyms and surface forms,
- negation patterns,
- disambiguation notes against adjacent symptoms.

It feeds:

- 2a, the embedding input per symptom,
- 2b, the BM25 synonym lexicon,
- 2c, the nearest-neighbor index,
- 4, the prompt context and the enum.

Build it once, derive every downstream artifact from it dynamically so there is no duplicated symptom data across components.

Schema:

```yaml
- label: fever
  definition_it: "Temperatura corporea elevata sopra la norma."
  examples_it:
    - "ha la febbre"
    - "scotta"
    - "temperatura alta"
    - "e bollente"
    - "ha 39 di febbre"
  synonyms_it: [febbre, temperatura, scotta, bollente]
  negation_patterns_it: ["non ha la febbre", "niente febbre", "la febbre e passata"]
  disambiguation: "Distinguere da sudorazione senza temperatura elevata."
  safety_critical: false

- label: nuchal_rigidity
  definition_it: "Rigidita del collo con resistenza alla flessione, possibile segno di meningite."
  examples_it:
    - "collo rigido"
    - "non riesce a piegare il collo"
    - "non vuole girare la testa e il collo e teso"
  synonyms_it: [collo rigido, rigidita nucale]
  negation_patterns_it: ["collo morbido", "muove bene il collo"]
  disambiguation: "Distinguere da semplice dolore muscolare al collo. La rigidita nucale implica resistenza alla flessione."
  safety_critical: true
```

Mine the catalog from your existing fine-tuning data. Each labeled example is a real Italian phrasing. Cluster by label, pick the 5 to 10 most diverse phrasings per symptom, and those become the examples. Safety-critical rare symptoms get more examples and explicit disambiguation notes.

## 6. Evaluation

Score each message as binary correct/wrong: correct only if the model got every symptom exactly right — nothing missed, nothing added. This matches the clinical requirement: a single missed symptom is a failed triage.

- Report accuracy (correct messages / total messages) as the primary metric.
- Stratify by case type: single-symptom, multi-symptom, negation, past-resolved.
- Track false negatives (missed symptoms) separately from false positives (extra symptoms) — in a clinical setting, misses are more dangerous than noise.
- Build the regression set from the existing fine-tuning data so it maps directly to the previous baseline.

## 7. Where to add or remove complexity

Order of operations when a symptom fails the floor:

1. Fix the catalog (more examples, better disambiguation). This resolves most implied-symptom and rare-symptom misses.
2. Fix the Stage 4 prompt (for example, instruct it to list every symptom even after finding several, to fix drops in long messages).
3. Add a pipeline stage (the per-symptom verifier, or a second extraction pass) only if 1 and 2 do not close the gap, and only for the failing symptoms.

What to drop: the separate verifier stage in the baseline. The constrained Stage 4 already verifies. Add it back only on measured precision loss.

## 8. Cost and operations

Per message in the full pipeline: roughly two frontier calls (Stage 2c is small, Stage 4 is the main one) plus cheap retrieval that is pure compute. Stage 1 can be rules. Stage 3 and Stage 5 are code. Cache the static catalog system prompt. Batch where possible. This lands cost-competitive with a hosted fine-tuned endpoint while staying future-proof, since adding a symptom to the catalog needs no retraining.

## 9. Optional strongest variant

If the dataset is large and clinical safety is paramount, run the pipeline above as the runtime system and also train a small open-weights model (Qwen2.5-7B or Llama-3.1-8B) with LoRA on the existing labeled data as a fallback. The pipeline handles new catalog symptoms without retraining; the small fine-tuned model handles the long-tail distribution. Ensemble by union with Stage 4 reconciliation. "Fine-tuning removed" usually means the managed endpoint, not that training your own is off the table.

## 10. Build order

1. Build the catalog YAML from the fine-tuning data.
2. Ship the single-call baseline with prompt caching.
3. Run it on the stratified test set, measure per-message accuracy.
4. For each symptom causing misses: catalog fix, then prompt fix, then pipeline addition, in that order.
5. Add Stage 2 retrieval and Stages 1, 3, 5 only for the slices the baseline cannot cover.
