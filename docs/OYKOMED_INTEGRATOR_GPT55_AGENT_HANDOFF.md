# GPT-5.5 Integrator Agent Handoff

## Mission

Implement the GPT-5.5 migration in the **Integrator repository** at:

```text
Oykomed-Integrator/oykomed
```

Replace these three existing AI-agent implementations:

| Agent | Existing method | Current fallback | Required result |
|---|---|---|---|
| Triage | `AiRoutingService.getUserQueryIntentAndSymptoms()` | Fine-tuned GPT-4.1 | GPT-5.5 constrained symptom extraction |
| General child care | `AiRoutingService.executeAdvisorModel()` | GPT-5.1 | GPT-5.5 advice using existing prompt and child context |
| Casual | `AiRoutingService.executeCasualModel()` | GPT-4.1 | GPT-5.5 streamed normal/refusal chat using existing prompt and context |

The implementation must preserve all caller-visible behavior. In particular, do **not** change TypeBot, FHIR persistence, route handlers, API contracts, or the router model in this task.

The complete file-by-file design and code skeleton is [OYKOMED_INTEGRATOR_GPT55_MIGRATION.md](OYKOMED_INTEGRATOR_GPT55_MIGRATION.md). Treat that document as the technical specification. This handoff defines how to execute it safely.

## Repository Boundary

There are two local codebases:

| Repository | Purpose | Commit implementation here? |
|---|---|---|
| Root `Oykomed` repository | GPT-5.5 evaluation, symptom catalog, migration documents | No implementation code |
| `Oykomed-Integrator/oykomed` | Production TypeScript server and three agent implementations | Yes |

Run `git status`, inspect the remote, create a feature branch, and commit from `Oykomed-Integrator/oykomed`. Do not add the Integrator directory to the root repository.

## Required Agent Skills

The implementation agent must be able to perform the following work:

| Skill | Required use |
|---|---|
| TypeScript and Node.js | Write strict TypeScript that compiles under `noImplicitAny`, `strictNullChecks`, and `noUnusedLocals` |
| Jest | Add focused unit tests with no live model calls |
| OpenAI Chat Completions tool calling | Use the existing OpenAI SDK transport through `CGPTUtils`; parse tool-call arguments defensively |
| Healthcare application safety | Preserve the existing triage, clinician escalation, and refusal behavior; never convert excluded symptoms into active symptoms |
| FHIR workflow awareness | Preserve `TriageFlowIdentification`, `VALID_SYMPTOMS`, `NEW_SYMPTOMS`, and existing case flow behavior without changing TypeBot |
| Secure logging | Never log raw parent messages, full prompts, API keys, child profiles, or tool payloads containing personal health data |
| Git hygiene | Commit only task-related Integrator files; do not revert unrelated user changes |

## Preconditions

Do these checks before modifying code:

```powershell
Set-Location Oykomed-Integrator/oykomed
git status --short
git remote -v
git branch --show-current
npm ci
```

If `npm ci` is inappropriate because the workspace intentionally uses an existing `node_modules`, document why and use the project-approved install command instead. Do not upgrade dependencies as part of this task.

Read these files before editing:

| File | Why it matters |
|---|---|
| `packages/server/src/oykomed/services/aiRoutingService.ts` | Contains all three implementation methods and the model-ID loader |
| `packages/server/src/oykomed/aiOperations/cgpt.ts` | Existing OpenAI client, retry, timeout, concurrency, metrics, and streaming behavior |
| `packages/server/src/oykomed/constants/aiPrompts.ts` | Existing general-child-care and casual safety prompts to preserve |
| `packages/server/src/oykomed/constants/symptomDefinitions.ts` | Clarification labels that must align with returned triage display labels |
| `packages/server/src/oykomed/type/typeBotIntegrationTypes.ts` | Existing `TriageFlowIdentification` contract that cannot change |
| `packages/server/src/oykomed/services/typeBotService.ts` | Read-only verification of downstream triage contract usage |
| `../../data/catalog/symptom_catalog.json` | Canonical evaluated symptom catalog to copy or generate from |
| `../../docs/EVALUATION_REPORT.md` | GPT-5.5 benchmark evidence and expected triage behavior |

## Immutable Contracts

Do not change the following during this task:

```ts
export type TriageFlowIdentification = {
  isTriage: boolean;
  symptoms: string[];
  newSymptoms: string[];
  needsSymptomDescription?: boolean;
  symptomDescriptionMessage?: string;
};
```

The three public method signatures must remain compatible:

```ts
getUserQueryIntentAndSymptoms(...): Promise<TriageFlowIdentification>
executeAdvisorModel(...): Promise<string>
executeCasualModel(...): Promise<string>
```

Do not modify these files unless a compile error proves an adjacent change is necessary:

```text
packages/server/src/oykomed/services/typeBotService.ts
packages/server/src/oykomed/handler/typeBotHandler.ts
packages/server/src/oykomed/type/typeBotIntegrationTypes.ts
packages/server/src/oykomed/aiOperations/cgpt.ts
```

Do not alter the router's intent classification, the TypeBot session flow, FHIR identifiers, AWS secret names, or response streaming protocol.

## Implementation Rules

### 1. Triage must use one source of truth

Create `packages/server/src/oykomed/constants/gpt55SymptomCatalog.ts` from the root catalog JSON. It must include all canonical entries and contain:

- stable `code`
- `labelIt`
- `labelEn`
- `triageDepth`
- `shortDefinition`

The model may return only a `code`. The application derives the display label using the local catalog. Never persist or return a model-supplied label without resolving it from the code.

### 2. Triage must have constrained structured output

Call GPT-5.5 through existing `this.cgptUtils.retryAI()` with a function tool that constrains the `code` field to the catalog-code enum. The payload must provide:

```ts
type Gpt55TriagePayload = {
  symptoms: Array<{
    code: string;
    evidenceSpan: string;
    negated: boolean;
    hedged: boolean;
    temporalStatus: 'current' | 'past_resolved' | 'chronic';
    confidence: 'high' | 'medium' | 'low';
    onset?: string | null;
  }>;
  excluded: Array<{
    code: string;
    reason: 'negated' | 'past_resolved' | 'below_threshold';
    evidenceSpan?: string | null;
  }>;
  unmappedComplaints: string[];
};
```

The agent must validate before adapting the result:

- `code` exists in the local catalog.
- `evidenceSpan` is a literal substring of the current user message.
- Duplicate codes are collapsed.
- `negated === true` and `temporalStatus === 'past_resolved'` never enter `symptoms`.
- Empty and whitespace-only unmapped complaints are removed.

Map active canonical symptoms to `TriageFlowIdentification.symptoms`, unmapped complaints to `newSymptoms`, and generic illness statements with no symptom or complaint to `needsSymptomDescription: true`.

The triage prompt receives only the current parent message. Do not send prior conversation history or the health summary to triage; the evaluated extraction design is current-message-only.

### 3. Preserve existing advice and casual behavior

For `executeAdvisorModel()` and `executeCasualModel()`, preserve existing:

- prompt content
- child profile construction
- conversation history injection
- advice response tool parsing
- `aiBypass` testing behavior
- casual normal/refusal modes
- photo guidance
- office-hours injection
- streaming callback and non-streaming fallback
- existing Italian fallback response on failure

Only change the selected fallback model to `gpt-5.5-2026-04-23`, and do not send a temperature parameter until the deployed GPT-5.5 API behavior has been verified. The model ID still comes first from `CUSTOM_AI_MODEL_NAMES`.

### 4. Keep secrets out of source control

Set this configuration outside Git in the existing `CUSTOM_AI_MODEL_NAMES` secret:

```json
{
  "TRIAGE_MODEL_ID": "gpt-5.5-2026-04-23",
  "ADVICE_MODEL_ID": "gpt-5.5-2026-04-23",
  "CASUAL_MODEL_ID": "gpt-5.5-2026-04-23"
}
```

Preserve the existing router and clarification fields. Set the legacy `VERTEX_TRIAGE_ENDPOINT` to an empty string when the GPT-5.5 triage code is deployed. Never print secret values, API keys, or raw SSM responses.

### 5. Resolve catalog drift before release

The new catalog and `SYMPTOM_EXPLANATIONS` differ. Resolve the following before enabling GPT-5.5 triage in production:

| New catalog | Legacy explanation map | Required action |
|---|---|---|
| Expanded `Poliuria` label | `Poliuria` | Adopt one display label and use it in both sources |
| Expanded `Pollachiuria` label | `Pollachiuria` | Adopt one display label and use it in both sources |
| `Terrore notturno` | Long legacy sleep-terror label | Adopt one canonical label and support the old text only as an alias if historical data needs it |
| Missing | `Feci molli` | Add to canonical catalog or formally exclude from GPT-5.5 triage scope |
| Missing | `Erezioni frequenti` | Add to canonical catalog or formally exclude from GPT-5.5 triage scope |

Do not silently drop a legacy symptom. If clinical ownership cannot make this decision, stop before production rollout and report the exact unresolved labels.

## Ordered Task List

Complete every task in order. After each code edit, run the focused test for that edit before expanding scope.

1. Create a feature branch in the Integrator repository, for example `feat/gpt55-agents`.
2. Add `gpt55SymptomCatalog.ts` with the full canonical catalog, prompt builder, constrained tool schema, parser, and adapter.
3. Add adapter unit tests. Run only those tests and fix failures before editing `AiRoutingService`.
4. Replace the body of `getUserQueryIntentAndSymptoms()` with the GPT-5.5 constrained tool call and adapter. Preserve its signature and `aiBypass` branch.
5. Add a mocked service test that verifies triage calls `retryAI()` with the configured model, a tool schema, `undefined` temperature, and a timeout.
6. Change the advisor fallback model to GPT-5.5 and replace its literal temperature arguments with `undefined`. Add a mocked test.
7. Change the casual fallback model to GPT-5.5 and replace its literal temperature arguments with `undefined`. Add streaming and fallback tests.
8. Reconcile `symptomDefinitions.ts` with the clinical-approved canonical display labels.
9. Run all required validation commands.
10. Provide a deployment change list for `CUSTOM_AI_MODEL_NAMES`; do not commit secrets.
11. Commit only source and test changes in the Integrator repository. Do not commit generated evaluation output, API keys, or unrelated changes.

## Required Tests

Use Jest and mock `CGPTUtils`. No test may call OpenAI, SSM, AWS, Vertex, or a database.

### Triage adapter tests

- Current explicit fever produces `['Febbre']`.
- Negated fever produces no active symptom.
- Past-resolved cough produces no active symptom.
- A missing evidence span is rejected.
- Duplicate codes produce one label.
- An unknown code is rejected.
- A generic illness message enables `needsSymptomDescription`.
- A non-catalog complaint becomes `newSymptoms`.
- The model cannot choose a code outside the tool enum.

### Service delegation tests

- Triage passes the system prompt and current user message only.
- Triage uses `TRIAGE_MODEL_ID` and falls back to `gpt-5.5-2026-04-23` only when it is absent.
- Advisor uses `ADVICE_MODEL_ID` and GPT-5.5 fallback.
- Casual uses `CASUAL_MODEL_ID` and GPT-5.5 fallback.
- Advisor keeps the `respond_parent_query` tool path and direct text fallback.
- Casual forwards streaming chunks in original order.
- Casual streaming failure falls back to non-streaming exactly once.

### Regression checks

- Existing `aiRoutingClassification.test.ts` remains green.
- Existing symptom-description flow tests remain green.
- Existing TypeBot tests remain green without source changes to TypeBot.

## Required Validation Commands

Run from `Oykomed-Integrator/oykomed`:

```powershell
npm run lint --workspace=@medplum/server
npm run build --workspace=@medplum/server
npm run test --workspace=@medplum/server -- gpt55Triage.test.ts aiRoutingClassification.test.ts aiRoutingSymptomDescription.test.ts
```

If workspace forwarding does not accept test-file arguments, run the package scripts directly:

```powershell
Set-Location packages/server
npm test -- gpt55Triage.test.ts aiRoutingClassification.test.ts aiRoutingSymptomDescription.test.ts
```

Run the standalone evaluation from the root `Oykomed` repository only after the deployed API key is available and billing is confirmed:

```powershell
python cli.py evaluate --provider openai --model gpt-5.5-2026-04-23 --concurrency 10 --output data/eval/eval_gpt55_integrator_baseline.json
```

The expected benchmark is 839/860 exact cases, or 97.6 percent. A lower result blocks triage promotion unless a clinical reviewer records why the prompt or catalog intentionally changed.

## Deployment Checklist

Before deployment:

- Confirm OpenAI endpoint and data-residency approval for pediatric data. Existing code routes GPT-5.5 to the non-EU OpenAI endpoint unless `CGPTUtils` is changed through a separately approved task.
- Confirm the GPT-5.5 model ID is available to the production OpenAI project.
- Confirm OpenAI account billing is active. `insufficient_quota` is a billing problem, not a retryable rate limit.
- Confirm catalog drift decisions are documented and applied.
- Confirm no raw PHI is included in new logs or tests.
- Confirm all required validation commands passed.

Deploy in this order:

1. Deploy code with old production model IDs unchanged.
2. Verify no behavior changes in server startup, router, TypeBot, and existing chat flow.
3. Switch only `TRIAGE_MODEL_ID` to GPT-5.5, restart the server, and run controlled staging messages.
4. Switch `ADVICE_MODEL_ID`, then verify child-context advice and direct text/tool responses.
5. Switch `CASUAL_MODEL_ID`, then verify normal, refusal, office-hours, and streaming responses.
6. Monitor timeout, parse-failure, empty-triage, and fallback metrics after each switch.

Rollback is a secret configuration change plus server restart:

```text
TRIAGE_MODEL_ID=<previous-fine-tuned-model-id>
ADVICE_MODEL_ID=<previous-advice-model-id>
CASUAL_MODEL_ID=<previous-casual-model-id>
```

No database rollback is expected because the external contracts and FHIR persistence format do not change.

## Definition of Done

The task is complete only when all of these are true:

- The Integrator implementation is committed in its own repository and branch.
- No TypeBot source file changed.
- `TriageFlowIdentification` did not change.
- Triage uses GPT-5.5 constrained codes, validates evidence spans, removes negated and resolved symptoms, and resolves labels locally.
- Advice and casual use GPT-5.5 through the existing `CGPTUtils` path.
- Existing prompt behavior, profile context, and streaming behavior are preserved for advice and casual.
- All required unit tests, lint, and build commands pass.
- The 860-message GPT-5.5 evaluation is attached or a documented blocker explains why it could not run.
- The deployment secret update and rollback model IDs are documented without exposing secret values.
- The final report lists changed files, executed commands, test results, model configuration changes, catalog decisions, residual risks, and rollback instructions.

## Expected Final Agent Report

The implementation agent must finish with this exact high-signal report structure:

```markdown
## Implemented
- Changed files:
- Triage behavior:
- Advice behavior:
- Casual behavior:

## Validation
- `npm run lint --workspace=@medplum/server`:
- `npm run build --workspace=@medplum/server`:
- Focused Jest tests:
- GPT-5.5 evaluation:

## Deployment Changes
- `CUSTOM_AI_MODEL_NAMES` fields to update:
- Restart/cache behavior:

## Catalog Decisions
- Final canonical labels:
- Legacy aliases retained:

## Residual Risks
- Data residency:
- API availability/billing:
- Any validation not run:

## Rollback
- Exact model-ID configuration changes:
```

## References

- [Detailed migration implementation specification](OYKOMED_INTEGRATOR_GPT55_MIGRATION.md)
- [Standalone GPT-5.5 evaluation report](EVALUATION_REPORT.md)
- [Standalone extraction architecture](SYSTEM_ARCHITECTURE.md)
- [Canonical symptom catalog](../data/catalog/symptom_catalog.json)
- [Integrator agent implementation](../Oykomed-Integrator/oykomed/packages/server/src/oykomed/services/aiRoutingService.ts)
- [Integrator OpenAI transport](../Oykomed-Integrator/oykomed/packages/server/src/oykomed/aiOperations/cgpt.ts)
- [Integrator prompts](../Oykomed-Integrator/oykomed/packages/server/src/oykomed/constants/aiPrompts.ts)
- [Integrator triage contract](../Oykomed-Integrator/oykomed/packages/server/src/oykomed/type/typeBotIntegrationTypes.ts)