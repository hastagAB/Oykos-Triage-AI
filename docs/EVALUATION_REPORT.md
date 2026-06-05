# Oykos Triage AI — Model Evaluation Report

**Date:** June 5, 2026
**Test dataset:** 860 Italian parent messages, covering 80 pediatric symptoms

---

## How We Score

Each test message has a known correct answer (the symptoms it contains). We run it through the full pipeline and check the output. A message is **correct** only if the model got every symptom exactly right — nothing missed, nothing added by mistake. Otherwise it's **wrong**.

That's the accuracy: **correct messages / total messages**.

---

## Results

| Model | Correct | Wrong | Accuracy |
|---|---|---|---|
| **GPT-5.5** (OpenAI, flagship) | **839 / 860** | 21 | **97.6%** |
| **Claude Sonnet 4.6** (Anthropic) | 826 / 860 | 34 | 96.0% |
| **Claude Opus 4** (Anthropic, flagship) | 823 / 860 | 37 | 95.7% |
| GPT-5.4 (OpenAI, fast) | 803 / 860 | 57 | 93.4% |
| GPT-5.4 Mini (OpenAI, budget) | 781 / 860 | 79 | 90.8% |
| GPT-5.4 Nano (OpenAI, cheapest) | 717 / 860 | 143 | 83.4% |

---

## OpenAI Models

### GPT-5.5 — 97.6% accurate (839 / 860 correct)

Best overall. Gets 97.6% of messages exactly right.

**21 mistakes:**
- 16 times: found all the right symptoms but also added an extra one that wasn't the main concern (e.g. a child with a toothache who can't sleep — it flagged "insomnia" as well as "dental pain", when the insomnia is just caused by the pain)
- 5 times: missed a symptom entirely

**Recommendation:** Best choice for production.

---

### GPT-5.4 — 93.4% accurate (803 / 860 correct)

**57 mistakes:**
- 42 times: added an extra symptom
- 13 times: missed a symptom
- 2 times: both missed one and added one wrong

Acceptable as a lower-cost alternative. More false alarms than GPT-5.5.

---

### GPT-5.4 Mini — 90.8% accurate (781 / 860 correct)

**79 mistakes:**
- 56 times: added an extra symptom
- 17 times: missed a symptom
- 6 times: both missed one and added one wrong

Usable for non-clinical applications. Not recommended where accuracy is critical.

---

### GPT-5.4 Nano — 83.4% accurate (717 / 860 correct)

**143 mistakes** — too many for clinical use.

- 78 times: added an extra symptom
- 36 times: missed a symptom (including common ones like constipation, toxic ingestion)
- 29 times: both missed one and added one wrong

**Not recommended for triage.**

---

## Anthropic Models

### Claude Sonnet 4.6 — 96.0% accurate (826 / 860 correct)

Best Anthropic model. Notably, it **outperforms the larger Opus 4** despite being a smaller model.

**34 mistakes:**
- 20 times: added an extra symptom
- 10 times: missed a symptom
- 4 times: both missed one and added one wrong

The mistakes it makes are similar in nature to GPT-5.5 — mostly borderline cases where two symptoms look alike (e.g. "nail discoloration" vs "nail fungus", "skin lesions" vs "insect bites").

**Recommendation:** Best Anthropic model. Safe for clinical use. Good choice if you want to avoid depending on a single provider.

---

### Claude Opus 4 — 95.7% accurate (823 / 860 correct)

Anthropic's flagship model, but it scores slightly below Sonnet 4.6 on this task.

**37 mistakes:**
- 18 times: missed a symptom (more misses than Sonnet)
- 17 times: added an extra symptom
- 2 times: both missed one and added one wrong

Notable weakness: struggles with seizure descriptions — missed "Convulsioni" (seizures) in 2 cases where the parent described the episode indirectly ("scatti ripetuti e irrigidimento", "scatti strani e non rispondeva").

**Recommendation:** Usable but Sonnet 4.6 is the better Anthropic choice for this task.

---

## What Both Top Models Get Right

- **Negations (100%):** When a parent says *"non ha la febbre"* (he does NOT have a fever), both models correctly ignore that symptom every single time.
- **Real production messages (100%):** Tested on 19 actual WhatsApp messages from parents — both GPT-5.5 and Claude Sonnet were perfect.
- **Multi-symptom messages:** Both handle messages with 2–3 symptoms cleanly.

---

## Common Mistake Pattern

The most frequent error across all models is **over-extraction** — the model finds a symptom that IS genuinely described in the message but is caused by the main symptom rather than being an independent concern.

Example: *"ha un dolore fortissimo al dente e non riesce a dormire"* (terrible toothache and can't sleep)
Expected: **Dental pain**
Wrong answer: **Dental pain + Insomnia** ← the insomnia is caused by the pain, not a separate symptom

This is the hardest problem to fix because the symptom IS present in the text — the model just needs to understand that it's a consequence, not an independent complaint.

---

## Recommendation

| Use case | Recommended model |
|---|---|
| Production triage | GPT-5.5 (OpenAI) |
| Provider independence / Anthropic | Claude Sonnet 4.6 |
| Cost-saving fallback | GPT-5.4 (OpenAI) |
| Avoid entirely | GPT-5.4 Nano |

GPT-5.5 handles **13 more messages correctly** than Claude Sonnet 4.6 out of 860 tests — a 1.6 percentage point gap. Both are safe for clinical use.

Surprisingly, **Claude Opus 4 scores lower than Sonnet 4.6** on this task (95.7% vs 96.0%). Bigger is not always better — Sonnet 4.6 is the better Anthropic choice here.

---

## Reproduce These Results

```bash
# Re-run evaluation for any model
python cli.py evaluate --provider openai --model gpt-5.5-2026-04-23 --output data/eval/results.json
python cli.py evaluate --provider anthropic --model claude-sonnet-4-6 --output data/eval/results.json
python cli.py evaluate --provider anthropic --model claude-opus-4-8 --output data/eval/results.json

# Show comparison across all saved results
python scripts/compare_results.py
```
