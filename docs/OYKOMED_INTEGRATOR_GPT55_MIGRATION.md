# Oykomed Integrator GPT-5.5 Agent Migration

> **For an implementation agent:** start with [OYKOMED_INTEGRATOR_GPT55_AGENT_HANDOFF.md](OYKOMED_INTEGRATOR_GPT55_AGENT_HANDOFF.md). It defines the repository boundary, immutable contracts, required skills, execution order, validation commands, deployment checklist, and final reporting format. Use this document as the detailed technical specification.

## Purpose

Replace the three existing parent-facing AI agents in `Oykomed-Integrator/oykomed` with specialist GPT-5.5 agents while keeping the router, TypeBot, FHIR case lifecycle, persistence, and client contracts unchanged.

The three agents are:

| Agent | Existing entry point | Existing fallback model | Replacement |
|---|---|---|---|
| Triage | `AiRoutingService.getUserQueryIntentAndSymptoms()` | Fine-tuned GPT-4.1 | Catalog-constrained GPT-5.5 extraction |
| General child care | `AiRoutingService.executeAdvisorModel()` | GPT-5.1 | GPT-5.5 profile-aware advice |
| Casual | `AiRoutingService.executeCasualModel()` | GPT-4.1 | GPT-5.5 streamed conversation and refusal handling |

The router remains a dispatcher. TypeBot remains downstream from the triage agent and is not changed by this migration.

```mermaid
flowchart LR
    M[Parent message] --> R[Existing router]
    R --> T[GPT-5.5 TriageAgent]
    R --> A[GPT-5.5 GeneralChildCareAgent]
    R --> C[GPT-5.5 CasualAgent]
    T --> B[Existing TypeBot flow]
    B --> F[Existing FHIR case workflow]
    A --> P[Existing chat persistence]
    C --> P
```

## Minimal Implementation Plan

This section is the implementation plan to follow. It deliberately avoids the larger `aiAgents/` refactor described later in this document.

The existing `AiRoutingService` already contains the three agents:

- `getUserQueryIntentAndSymptoms()` is the triage agent.
- `executeAdvisorModel()` is the general child care agent.
- `executeCasualModel()` is the casual agent.

Therefore, the first production change needs only one new TypeScript file, three focused tests, and edits to the existing routing service. Do not change `TypeBotService`.

### Exact Files To Change

| Step | File | Change | Why |
|---|---|---|---|
| 1 | `packages/server/src/oykomed/constants/gpt55SymptomCatalog.ts` | Add canonical symptom catalog, prompt builder, tool schema, and result parser | Gives the triage agent one local source of truth |
| 2 | `packages/server/src/oykomed/services/aiRoutingService.ts` | Import the helper and replace only `getUserQueryIntentAndSymptoms()` | Replaces the fine-tuned triage model without changing callers |
| 3 | `packages/server/src/oykomed/services/aiRoutingService.ts` | Change advice fallback model to GPT-5.5 and omit temperature | Uses GPT-5.5 while retaining the established prompt and profile assembly |
| 4 | `packages/server/src/oykomed/services/aiRoutingService.ts` | Change casual fallback model to GPT-5.5 and omit temperature | Uses GPT-5.5 while retaining normal/refusal modes and streaming |
| 5 | `packages/server/src/oykomed/constants/symptomDefinitions.ts` | Reconcile from the new catalog only after clinical review | Keeps clarification text aligned with triage labels |
| 6 | `packages/server/src/oykomed/services/gpt55Triage.test.ts` | Add parsing and adapter tests | Prevents invalid symptoms from reaching cases |
| 7 | `packages/server/src/oykomed/services/aiRoutingService.test.ts` or existing focused tests | Add advice and casual model-selection tests | Verifies the existing methods select GPT-5.5 |
| 8 | Project secret `CUSTOM_AI_MODEL_NAMES` | Point three agent model IDs at GPT-5.5 | Enables the deployed model without hard-coding it |

No other source file is required for the first implementation.

### Files Explicitly Not Changed

Do not edit any of these files in the first implementation:

| File | Reason |
|---|---|
| `packages/server/src/oykomed/services/typeBotService.ts` | It already consumes `TriageFlowIdentification`; preserving that return type avoids workflow changes |
| `packages/server/src/oykomed/type/typeBotIntegrationTypes.ts` | `TriageFlowIdentification` remains unchanged |
| `packages/server/src/oykomed/handler/typeBotHandler.ts` | API request and response behavior remain unchanged |
| `packages/server/src/oykomed/aiOperations/cgpt.ts` | Reuse its SSM credentials, rate limit gate, retry, timeout, logging, and streaming |
| `packages/server/src/oykomed/services/aiRoutingService.ts` router methods | The router is not one of the three replacements |

## Step-by-Step Code Changes

### Step 1: Add the Canonical Triage Catalog Helper

**Create:** `packages/server/src/oykomed/constants/gpt55SymptomCatalog.ts`

Copy the contents of the evaluated source catalog, [data/catalog/symptom_catalog.json](../data/catalog/symptom_catalog.json), into this file as TypeScript data. Do not copy the legacy `SYMPTOM_EXPLANATIONS` map as the triage source of truth.

Start with this exact structure:

```ts
import { TriageFlowIdentification } from '../type/typeBotIntegrationTypes';

export type CatalogSymptom = {
  code: string;
  labelIt: string;
  labelEn: string;
  triageDepth: string;
  shortDefinition: string;
};

export type Gpt55TriageItem = {
  code: string;
  evidenceSpan: string;
  negated: boolean;
  hedged: boolean;
  temporalStatus: 'current' | 'past_resolved' | 'chronic';
  confidence: 'high' | 'medium' | 'low';
  onset?: string | null;
};

export type Gpt55TriagePayload = {
  symptoms: Gpt55TriageItem[];
  excluded: Array<{
    code: string;
    reason: 'negated' | 'past_resolved' | 'below_threshold';
    evidenceSpan?: string | null;
  }>;
  unmappedComplaints: string[];
};

export const GPT55_TRIAGE_MODEL = 'gpt-5.5-2026-04-23';

export const SYMPTOM_CATALOG: readonly CatalogSymptom[] = [
  {
    code: 'SI001',
    labelIt: 'Febbre',
    labelEn: 'Fever',
    triageDepth: 'Alta',
    shortDefinition: 'Aumento della temperatura corporea rispetto al normale.',
  },
  // Add every remaining entry from data/catalog/symptom_catalog.json here.
];

const SYMPTOM_BY_CODE = new Map(
  SYMPTOM_CATALOG.map((symptom) => [symptom.code, symptom])
);

const GENERIC_ILLNESS_PATTERN =
  /\b(non\s+sta\s+bene|non\s+si\s+sente\s+bene|sta\s+male|è\s+malat[oa]|è\s+ammalat[oa])\b/i;
```

The final implementation must include all catalog entries. The small example above exists only to show the file shape.

Add a prompt builder below the catalog:

```ts
export function buildGpt55TriageSystemPrompt(): string {
  const catalogText = SYMPTOM_CATALOG.map(
    (symptom) => [
      `Codice: ${symptom.code}`,
      `Etichetta: ${symptom.labelIt}`,
      `Definizione: ${symptom.shortDefinition}`,
    ].join('\n')
  ).join('\n\n');

  return `Sei un estrattore di sintomi pediatrici da messaggi di genitori italiani.

Estrai tutti i sintomi attualmente presenti dal messaggio.
Usa esclusivamente i codici del catalogo.
Non includere sintomi negati, per esempio "non ha la febbre".
Non includere sintomi risolti, per esempio "la tosse è passata".
Per ogni sintomo estratto, evidenceSpan deve essere una sottostringa letterale del messaggio del genitore.
Quando un sintomo è incerto, includilo con confidence "medium" o "low" e hedged true.
Quando una lamentela non appartiene al catalogo, inserisci le parole del genitore in unmappedComplaints.

CATALOGO SINTOMI
${catalogText}`;
}
```

Create the constrained tool schema in the same file. Constrain only the code. The application derives every label from the code after parsing.

```ts
export const GPT55_TRIAGE_TOOL = {
  type: 'function' as const,
  function: {
    name: 'extract_pediatric_symptoms',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        symptoms: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              code: { type: 'string', enum: SYMPTOM_CATALOG.map((symptom) => symptom.code) },
              evidenceSpan: { type: 'string' },
              negated: { type: 'boolean' },
              hedged: { type: 'boolean' },
              temporalStatus: { type: 'string', enum: ['current', 'past_resolved', 'chronic'] },
              confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
              onset: { type: ['string', 'null'] },
            },
            required: ['code', 'evidenceSpan', 'negated', 'hedged', 'temporalStatus', 'confidence'],
          },
        },
        excluded: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              code: { type: 'string', enum: SYMPTOM_CATALOG.map((symptom) => symptom.code) },
              reason: { type: 'string', enum: ['negated', 'past_resolved', 'below_threshold'] },
              evidenceSpan: { type: ['string', 'null'] },
            },
            required: ['code', 'reason'],
          },
        },
        unmappedComplaints: {
          type: 'array',
          items: { type: 'string' },
        },
      },
      required: ['symptoms', 'excluded', 'unmappedComplaints'],
    },
  },
};
```

Finally, add the adapter that protects the existing TypeBot contract:

```ts
export function toTriageFlowIdentification(
  message: string,
  payload: Gpt55TriagePayload
): TriageFlowIdentification {
  const symptomCodes = new Set<string>();
  const symptoms: string[] = [];

  for (const item of payload.symptoms) {
    const catalogSymptom = SYMPTOM_BY_CODE.get(item.code);
    const isExcluded = item.negated || item.temporalStatus === 'past_resolved';
    const hasValidEvidence = message.includes(item.evidenceSpan);

    if (!catalogSymptom || isExcluded || !hasValidEvidence || symptomCodes.has(item.code)) {
      continue;
    }

    symptomCodes.add(item.code);
    symptoms.push(catalogSymptom.labelIt);
  }

  const newSymptoms = payload.unmappedComplaints
    .map((complaint) => complaint.trim())
    .filter(Boolean);
  const needsSymptomDescription =
    symptoms.length === 0 && newSymptoms.length === 0 && GENERIC_ILLNESS_PATTERN.test(message);

  return {
    isTriage: symptoms.length > 0 || newSymptoms.length > 0 || needsSymptomDescription,
    symptoms,
    newSymptoms,
    needsSymptomDescription,
    symptomDescriptionMessage: needsSymptomDescription
      ? 'Per favore, prova a descrivermi con le tue parole uno o più sintomi che il bambino sta mostrando.'
      : '',
  };
}
```

This helper is the only new production file required for the first triage release.

### Step 2: Replace Only the Triage Method Body

**Edit:** `packages/server/src/oykomed/services/aiRoutingService.ts`

At the import block, add:

```ts
import {
  buildGpt55TriageSystemPrompt,
  GPT55_TRIAGE_MODEL,
  GPT55_TRIAGE_TOOL,
  Gpt55TriagePayload,
  toTriageFlowIdentification,
} from '../constants/gpt55SymptomCatalog';
```

Then replace the complete body of the existing `getUserQueryIntentAndSymptoms()` method. Do not rename the method and do not modify its call sites.

```ts
async getUserQueryIntentAndSymptoms(
  userQuery: string,
  _aiProvider: string,
  _previousConversation: string,
  _childrenHealthSummary: string,
  timeoutMs?: number,
  aiBypass?: AiBypassData
): Promise<TriageFlowIdentification> {
  if (aiBypass?.intent === AI_ROUTER_MODEL_OUTPUTS.TRIAGE && aiBypass.symptoms) {
    return {
      isTriage: true,
      symptoms: aiBypass.symptoms,
      newSymptoms: [],
    };
  }

  try {
    const configuredModel = (await this.getAiCustomModelIDs()).triageModelId;
    const messages: unknown[] = [];
    this.cgptUtils.pushMessage('system', buildGpt55TriageSystemPrompt(), messages);
    this.cgptUtils.pushMessage('user', userQuery, messages);

    const response = await this.cgptUtils.retryAI(
      messages,
      configuredModel || GPT55_TRIAGE_MODEL,
      [GPT55_TRIAGE_TOOL],
      'auto',
      undefined,
      timeoutMs ?? DEFAULT_AI_TIMEOUT_MS,
      1024
    );

    const toolArguments = response?.choices?.[0]?.message?.tool_calls?.[0]?.function?.arguments;
    if (!toolArguments) {
      throw new Error('GPT-5.5 triage response did not include tool arguments.');
    }

    const payload = JSON.parse(toolArguments) as Gpt55TriagePayload;
    return toTriageFlowIdentification(userQuery, payload);
  } catch (error) {
    logError('Error getting GPT-5.5 triage result', error, {
      operation: 'getUserQueryIntentAndSymptoms',
    });
    errorTracker.trackError(error, {
      service: 'ai-routing',
      operation: 'getUserQueryIntentAndSymptoms',
    });
    return { isTriage: false, symptoms: [], newSymptoms: [] };
  }
}
```

Delete the old triage-only code inside this method after the replacement:

- Loading `BOT_SYMPTOMS_LIST` from SSM.
- Merging `SYMPTOM_EXPLANATIONS` into the runtime symptom list.
- The `AI_PROVIDER.AMAZON` branch.
- Vertex triage endpoint fallback.
- The old fine-tuned fallback model `ft:gpt-4.1-2025-04-14:oykomed:oykomed-symptom-4-1:CdOLqMzI`.
- The old `extract_symptoms` tool schema.
- Triage conversation and health-summary prompt injection.

Do not delete `BedrockUtil`, Vertex imports, or router logic elsewhere in the file. They may still be used by other paths until a separate cleanup task verifies they are unused.

### Step 3: Switch the General Child Care Agent

**Edit:** `packages/server/src/oykomed/services/aiRoutingService.ts`

The existing `executeAdvisorModel()` method already builds the correct parent conversation and child profile context. Do not rewrite the prompt or the profile-building logic initially.

Make only these changes:

```ts
// Before
const advisorModelId = customAIModelsID.adviceModelId || 'gpt-5.1-2025-11-13';

// After
const advisorModelId = customAIModelsID.adviceModelId || GPT55_TRIAGE_MODEL;
```

Replace every literal temperature `0` in the three advisor execution calls with `undefined`:

```ts
// Before
await this.cgptUtils.retryAI(messageList, advisorModelId, tools, 'auto', 0);

// After
await this.cgptUtils.retryAI(messageList, advisorModelId, tools, 'auto', undefined);
```

Use the same replacement for `executeAIStreamWithTools()`. Keep:

- `CUSTOM_MODEL_AI_PROMPTS.advisorModel`.
- The child profile and health-summary logic.
- The `respond_parent_query` tool.
- Existing direct-text fallback parsing.
- The existing `aiBypass` behavior.

The advice method must still return `Promise<string>`. Nothing calling it changes.

### Step 4: Switch the Casual Agent

**Edit:** `packages/server/src/oykomed/services/aiRoutingService.ts`

The existing `executeCasualModel()` already owns normal mode, refusal mode, photo guidance, office hours, profile context, and streaming. Do not move this behavior into a new class in the first implementation.

Make only these changes:

```ts
// Before
const casualModelId = customAIModelsID.casualModelId || 'gpt-4.1-2025-04-14';

// After
const casualModelId = customAIModelsID.casualModelId || GPT55_TRIAGE_MODEL;
```

Change the temperature argument in all three casual calls from `0.7` to `undefined`:

```ts
// Streaming
for await (const chunk of this.cgptUtils.executeAIStream(
  messageList,
  casualModelId,
  undefined,
  undefined,
  undefined,
  DEFAULT_AI_TIMEOUT_MS,
  casualModelMaxOutputTokens
)) {
  response += chunk;
  onChunk(chunk);
}

// Non-streaming and fallback
aiResponse = await this.cgptUtils.retryAI(
  messageList,
  casualModelId,
  undefined,
  undefined,
  undefined,
  DEFAULT_AI_TIMEOUT_MS,
  casualModelMaxOutputTokens
);
```

Keep the existing `try/catch` around streaming and its non-streaming fallback. The method must still return its current Italian failure message when both paths fail.

### Step 5: Update the Deployed Model Configuration

**Edit outside source control:** the value of `PROJECT_SECRETS.CUSTOM_AI_MODEL_NAMES` in the Integrator project configuration.

Use this JSON payload, preserving the existing router, clarification, and Vertex fields:

```json
{
  "ROUTER_MODEL_ID": "<existing value>",
  "TRIAGE_MODEL_ID": "gpt-5.5-2026-04-23",
  "ADVICE_MODEL_ID": "gpt-5.5-2026-04-23",
  "CLARIFICATION_MODEL_ID": "<existing value>",
  "CASUAL_MODEL_ID": "gpt-5.5-2026-04-23",
  "VERTEX_ROUTER_ENDPOINT": "<existing value>",
  "VERTEX_TRIAGE_ENDPOINT": ""
}
```

Set `VERTEX_TRIAGE_ENDPOINT` to an empty string for the new triage path. Otherwise the legacy Vertex branch remains configured even though the implementation no longer uses it.

`getAiCustomModelIDs()` caches this secret for five minutes. Restart the server after changing it, or wait for the cache expiry before verifying the selected model.

### Step 6: Reconcile Clarification Labels

**Edit after triage tests pass:** `packages/server/src/oykomed/constants/symptomDefinitions.ts`

This map is used for symptom explanations and clarification buttons. It is not the source of triage truth after Step 2, but it must contain every display label returned by `toTriageFlowIdentification()`.

Make these decisions with clinical ownership before editing:

| New catalog value | Legacy value | Required decision |
|---|---|---|
| `Poliuria (emissione di abbondante quantità di urina)` | `Poliuria` | Choose one display label, update catalog and explanation key together |
| `Pollachiuria (necessità di urinare molto spesso ma con piccole quantità)` | `Pollachiuria` | Choose one display label, update catalog and explanation key together |
| `Terrore notturno` | `Urla nel sonno in stato di semi incoscienza e forte agitazione` | Keep one clinically approved label and map the other only as a legacy alias |
| No entry | `Feci molli` | Add it to the canonical catalog or explicitly treat it as non-canonical |
| No entry | `Erezioni frequenti` | Add it to the canonical catalog or explicitly treat it as non-canonical |

Do not release the triage change while these labels are unresolved. A returned triage label without an explanation map entry breaks the expected clarification experience.

### Step 7: Add Tests Before Enabling GPT-5.5

**Create:** `packages/server/src/oykomed/services/gpt55Triage.test.ts`

The adapter can be tested without network access. Start with these exact tests:

```ts
import { toTriageFlowIdentification } from '../constants/gpt55SymptomCatalog';

describe('toTriageFlowIdentification', () => {
  it('keeps a catalog symptom with verbatim evidence', () => {
    const result = toTriageFlowIdentification('Mio figlio ha la febbre alta', {
      symptoms: [{
        code: 'SI001',
        evidenceSpan: 'ha la febbre alta',
        negated: false,
        hedged: false,
        temporalStatus: 'current',
        confidence: 'high',
      }],
      excluded: [],
      unmappedComplaints: [],
    });

    expect(result).toEqual({
      isTriage: true,
      symptoms: ['Febbre'],
      newSymptoms: [],
      needsSymptomDescription: false,
      symptomDescriptionMessage: '',
    });
  });

  it('does not include a negated symptom', () => {
    const result = toTriageFlowIdentification('Non ha la febbre', {
      symptoms: [{
        code: 'SI001',
        evidenceSpan: 'Non ha la febbre',
        negated: true,
        hedged: false,
        temporalStatus: 'current',
        confidence: 'high',
      }],
      excluded: [],
      unmappedComplaints: [],
    });

    expect(result.symptoms).toEqual([]);
    expect(result.isTriage).toBe(false);
  });

  it('asks for description after a generic illness statement', () => {
    const result = toTriageFlowIdentification('Mio figlio non sta bene', {
      symptoms: [],
      excluded: [],
      unmappedComplaints: [],
    });

    expect(result).toMatchObject({
      isTriage: true,
      needsSymptomDescription: true,
    });
  });
});
```

Add three more tests:

1. An evidence span absent from the parent message does not become a symptom.
2. The same code twice produces one display label.
3. A non-catalog complaint appears in `newSymptoms`.

**Edit:** `packages/server/src/oykomed/services/aiRoutingClassification.test.ts`

Add small unit tests with `CGPTUtils` mocked that assert:

```ts
expect(mockRetryAI).toHaveBeenCalledWith(
  expect.any(Array),
  'gpt-5.5-2026-04-23',
  expect.anything(),
  expect.anything(),
  undefined,
  expect.any(Number),
  expect.any(Number)
);
```

Use one test for advice and one test for casual. Do not make live OpenAI calls from Jest.

### Step 8: Run and Release

From `Oykomed-Integrator/oykomed`:

```powershell
npm run build --workspace=@medplum/server
npm run test --workspace=@medplum/server -- gpt55Triage.test.ts aiRoutingClassification.test.ts
```

From the root extraction repository:

```powershell
python cli.py evaluate --provider openai --model gpt-5.5-2026-04-23 --concurrency 10 --output data/eval/eval_gpt55_integrator_baseline.json
```

Release in this order:

1. Deploy the source code with all three `*_MODEL_ID` values still set to the old values. Confirm server startup and existing traffic are unchanged.
2. Change only `TRIAGE_MODEL_ID` to `gpt-5.5-2026-04-23`. Confirm the structured tool response and `TriageFlowIdentification` for a test parent account.
3. Check negated, resolved, multi-symptom, generic-illness, and unmapped-complaint messages in a staging environment.
4. Change `ADVICE_MODEL_ID` to GPT-5.5. Check a normal advice reply with a child profile and conversation history.
5. Change `CASUAL_MODEL_ID` to GPT-5.5. Check normal mode, refusal mode, office-hours response, and streamed response.
6. Keep the previous model IDs in the deployment runbook. Rollback is only a secret change plus server restart.

### Why This Is the Minimal Path

The initial implementation does not need a new agent framework because the current code already separates the three agent entry points. The only behavior that must fundamentally change is triage extraction. Advice and casual can use GPT-5.5 by changing their model selection while retaining their mature prompt and integration logic.

Refactor into separate `aiAgents/` classes only after the GPT-5.5 behavior is stable and measured in production.

## Evidence and Scope

The standalone extraction system uses a single GPT-5.5 call with a full symptom catalog and constrained structured output. Its documented evaluation result is 839 correct messages out of 860, or 97.6 percent exact-message accuracy.

References in this repository:

- [Standalone system overview](../README.md)
- [Extraction architecture](SYSTEM_ARCHITECTURE.md)
- [GPT-5.5 evaluation](EVALUATION_REPORT.md)
- [Canonical symptom catalog](../data/catalog/symptom_catalog.json)
- [Existing Integrator routing service](../Oykomed-Integrator/oykomed/packages/server/src/oykomed/services/aiRoutingService.ts)
- [Existing Integrator prompts](../Oykomed-Integrator/oykomed/packages/server/src/oykomed/constants/aiPrompts.ts)

`Oykomed-Integrator` is currently an untracked local directory in this repository. The Integrator code changes must be committed in its own repository or added deliberately in a separate change. This document is a design and implementation plan only.

## Non-Goals

- Do not change TypeBot questions, sessions, or case progression.
- Do not change FHIR `Communication` identifiers or the existing `TriageFlowIdentification` shape.
- Do not replace the router model in this migration.
- Do not deploy the optional Python five-stage retrieval pipeline. The evaluated one-call catalog baseline is the production target.
- Do not remove the existing `CGPTUtils` retry, timeout, concurrency, logging, or streaming facilities.

## Target Layout

Create a small, explicit agent layer under the Integrator server package:

```text
packages/server/src/oykomed/aiAgents/
  agentModelConfig.ts          # GPT-5.5 model IDs and migration mode
  gpt55AgentClient.ts          # Shared Chat Completions wrapper over CGPTUtils
  prompts.ts                   # Triage, advice, and casual prompts
  triage/
    symptomCatalog.ts          # Canonical symptom entries keyed by code
    triageAgent.ts             # Catalog constrained extraction
    triageAdapter.ts           # TriageAgentResult -> TriageFlowIdentification
  generalChildCareAgent.ts     # Profile-aware general guidance
  casualAgent.ts               # Normal/refusal conversation with streaming
  types.ts                     # Agent-only request and response types
```

Keep `AiRoutingService` as the orchestration boundary. It should construct the three agents once and delegate existing public methods to them.

## Phase 0: Decide and Freeze the Canonical Catalog

Do this before implementation. It prevents a model result from being correctly extracted but failing downstream confirmation or clarification behavior.

Current local comparison found:

| Catalog source | Label count |
|---|---:|
| New extraction catalog | 81 |
| Legacy `SYMPTOM_EXPLANATIONS` map | 82 |
| Exact label overlap | 77 |

The notable differences are:

- The new catalog uses expanded labels for `Poliuria` and `Pollachiuria`.
- The new catalog contains `Terrore notturno`.
- The legacy map contains `Feci molli`, `Erezioni frequenti`, and a long legacy sleep-terror label.

### Decision

Use the new catalog's `code` as the canonical integration identifier. A catalog entry must include both the stable code and the display label:

```ts
export type SymptomCatalogEntry = {
  code: string;
  labelIt: string;
  labelEn: string;
  triageDepth: 'Alta' | 'Media';
  shortDefinition: string;
};

export const symptomCatalog: readonly SymptomCatalogEntry[] = [
  {
    code: 'SI001',
    labelIt: 'Febbre',
    labelEn: 'Fever',
    triageDepth: 'Alta',
    shortDefinition: 'Aumento della temperatura corporea rispetto al normale.',
  },
];

export const symptomByCode = new Map(
  symptomCatalog.map((symptom) => [symptom.code, symptom])
);
```

The implementation must never trust a model-provided label. It accepts only a code and derives `labelIt` from `symptomByCode`.

Before production rollout, reconcile all 81 or 82 entries with clinical product ownership and generate `symptomCatalog.ts` from `data/catalog/symptom_catalog.json`. Do not maintain two manually edited symptom lists.

## Phase 1: Add Shared GPT-5.5 Configuration

### 1. Configure existing model-secret fields

Keep the current `CUSTOM_AI_MODEL_NAMES` secret shape. Update only the three agent model IDs:

```json
{
  "ROUTER_MODEL_ID": "<keep-existing-router-model>",
  "TRIAGE_MODEL_ID": "gpt-5.5-2026-04-23",
  "ADVICE_MODEL_ID": "gpt-5.5-2026-04-23",
  "CASUAL_MODEL_ID": "gpt-5.5-2026-04-23"
}
```

Do not change `ROUTER_MODEL_ID` in this migration. The router decides which of the three specialist agents runs.

### 2. Add explicit migration mode

Add a non-secret `AI_AGENT_MIGRATION_MODE` setting with these values:

| Value | Behavior |
|---|---|
| `legacy` | Current implementation only |
| `shadow` | Run the GPT-5.5 agent, log a redacted comparison, return the legacy result |
| `enabled` | Return the GPT-5.5 agent result |

The default must be `legacy`. Enable agents independently so triage can be released before advice and casual.

```ts
export type AgentMigrationMode = 'legacy' | 'shadow' | 'enabled';

export function getAgentMigrationMode(agent: 'triage' | 'advice' | 'casual'): AgentMigrationMode {
  const value = process.env[`AI_${agent.toUpperCase()}_AGENT_MODE`] || 'legacy';
  return value === 'shadow' || value === 'enabled' ? value : 'legacy';
}
```

### 3. Confirm data residency before enabling

`CGPTUtils` currently uses the EU endpoint only for `gpt-4.1-2025-04-14`; GPT-5.5 would currently use the US endpoint. Confirm the approved endpoint, data-processing agreement, retention settings, and request logging policy for pediatric data before moving any agent to `enabled`.

## Phase 2: Implement the Shared Client

Use the existing `CGPTUtils` transport. It already provides SSM-backed credentials, bounded concurrency, retries, metrics, timeout handling, and streaming. The agent layer only standardizes model choice and parsing.

```ts
import { CGPTUtils, DEFAULT_AI_TIMEOUT_MS } from '../aiOperations/cgpt';

export const GPT55_MODEL_ID = 'gpt-5.5-2026-04-23';

export class Gpt55AgentClient {
  public constructor(private readonly cgptUtils = new CGPTUtils()) {}

  public async complete(
    systemPrompt: string,
    userPrompt: string,
    options: {
      model: string;
      tools?: unknown[];
      toolChoice?: 'auto';
      maxTokens?: number;
    }
  ): Promise<unknown> {
    const messages: unknown[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ];

    return this.cgptUtils.retryAI(
      messages,
      options.model,
      options.tools,
      options.toolChoice,
      undefined,
      DEFAULT_AI_TIMEOUT_MS,
      options.maxTokens
    );
  }
}
```

Use `undefined` for temperature in the shared client. The triage agent needs deterministic structured extraction, and model parameters must be set per agent only after confirming they are supported by GPT-5.5.

Do not create a second OpenAI SDK client or duplicate the existing request retry logic.

## Phase 3: Implement the Triage Agent

### Public contract

Keep the current Integrator contract unchanged:

```ts
export type TriageFlowIdentification = {
  isTriage: boolean;
  symptoms: string[];
  newSymptoms: string[];
  needsSymptomDescription?: boolean;
  symptomDescriptionMessage?: string;
};
```

Internally, use a richer code-first result:

```ts
export type ExtractedSymptom = {
  code: string;
  evidenceSpan: string;
  negated: boolean;
  hedged: boolean;
  temporalStatus: 'current' | 'past_resolved' | 'chronic';
  confidence: 'high' | 'medium' | 'low';
  onset?: string;
};

export type ExcludedSymptom = {
  code: string;
  reason: 'negated' | 'past_resolved' | 'below_threshold';
  evidenceSpan?: string;
};

export type TriageAgentResult = {
  symptoms: ExtractedSymptom[];
  excluded: ExcludedSymptom[];
  unmappedComplaints: string[];
};
```

### Structured tool schema

Constrain symptom codes in the tool schema and derive labels locally. Do not use an independent label enum.

```ts
const symptomCodes = symptomCatalog.map((symptom) => symptom.code);

const triageTool = {
  type: 'function' as const,
  function: {
    name: 'extract_pediatric_symptoms',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        symptoms: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              code: { type: 'string', enum: symptomCodes },
              evidenceSpan: { type: 'string' },
              negated: { type: 'boolean' },
              hedged: { type: 'boolean' },
              temporalStatus: {
                type: 'string',
                enum: ['current', 'past_resolved', 'chronic'],
              },
              confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
              onset: { type: ['string', 'null'] },
            },
            required: ['code', 'evidenceSpan', 'negated', 'hedged', 'temporalStatus', 'confidence'],
          },
        },
        excluded: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              code: { type: 'string', enum: symptomCodes },
              reason: { type: 'string', enum: ['negated', 'past_resolved', 'below_threshold'] },
              evidenceSpan: { type: ['string', 'null'] },
            },
            required: ['code', 'reason'],
          },
        },
        unmappedComplaints: {
          type: 'array',
          items: { type: 'string' },
        },
      },
      required: ['symptoms', 'excluded', 'unmappedComplaints'],
    },
  },
};
```

### Post-processing and adapter

Validate every code against `symptomByCode`; verify every evidence span is a substring of the original parent message; remove duplicate codes; and move negated or past-resolved symptoms to `excluded` even if the model placed them in `symptoms`.

```ts
const GENERIC_ILLNESS = /\b(non\s+sta\s+bene|non\s+si\s+sente\s+bene|sta\s+male|è\s+malat[oa]|è\s+ammalat[oa])\b/i;

export function toTriageFlowIdentification(
  message: string,
  result: TriageAgentResult
): TriageFlowIdentification {
  const symptoms = result.symptoms
    .filter((symptom) => !symptom.negated && symptom.temporalStatus !== 'past_resolved')
    .map((symptom) => symptomByCode.get(symptom.code)?.labelIt)
    .filter((label): label is string => Boolean(label));

  const needsSymptomDescription =
    symptoms.length === 0 && result.unmappedComplaints.length === 0 && GENERIC_ILLNESS.test(message);

  return {
    isTriage: symptoms.length > 0 || result.unmappedComplaints.length > 0 || needsSymptomDescription,
    symptoms,
    newSymptoms: result.unmappedComplaints,
    needsSymptomDescription,
    symptomDescriptionMessage: needsSymptomDescription
      ? 'Per favore, prova a descrivermi con le tue parole uno o più sintomi che il bambino sta mostrando.'
      : '',
  };
}
```

This preserves current TypeBot behavior without changing TypeBot itself:

- `symptoms` continues to populate `VALID_SYMPTOMS` and the FHIR case topic.
- `newSymptoms` continues to populate `NEW_SYMPTOMS` and trigger its existing notification behavior.
- `needsSymptomDescription` continues to use the existing guidance loop.
- `excluded` is not treated as an active symptom. Log only a redacted audit record containing codes and reasons.

### Delegation from `AiRoutingService`

Replace only the internals of `getUserQueryIntentAndSymptoms()`. Keep the method name and arguments until all current call sites are migrated.

```ts
async getUserQueryIntentAndSymptoms(
  userQuery: string,
  _aiProvider: string,
  _previousConversation: string,
  _childrenHealthSummary: string,
  _timeoutMs?: number,
  aiBypass?: AiBypassData
): Promise<TriageFlowIdentification> {
  if (aiBypass?.intent === AI_ROUTER_MODEL_OUTPUTS.TRIAGE && aiBypass.symptoms) {
    return { isTriage: true, symptoms: aiBypass.symptoms, newSymptoms: [] };
  }

  const result = await this.triageAgent.extract(userQuery);
  return toTriageFlowIdentification(userQuery, result);
}
```

Do not pass previous conversation or health summary into the baseline symptom extractor. The evaluated extraction task operates on the current parent message, and unrelated history can create false positives.

## Phase 4: Implement the General Child Care Agent

The general child care agent remains a text-response agent. Keep the existing child profile construction and the current Italian safety policy from `CUSTOM_MODEL_AI_PROMPTS.advisorModel`.

Create a small request type:

```ts
export type GeneralChildCareRequest = {
  userQuery: string;
  chatHistory: string;
  childContext: string;
};

export class GeneralChildCareAgent {
  public constructor(
    private readonly client: Gpt55AgentClient,
    private readonly modelId: string
  ) {}

  public async respond(request: GeneralChildCareRequest): Promise<string> {
    const response = await this.client.complete(
      GENERAL_CHILD_CARE_SYSTEM_PROMPT,
      buildGeneralChildCareUserPrompt(request),
      { model: this.modelId, maxTokens: 1024 }
    );

    return parseResponseToolOrText(response);
  }
}
```

Retain the existing `respond_parent_query` tool only if GPT-5.5 requires it for reliable output. Otherwise use direct text output. The public `executeAdvisorModel()` contract must remain `Promise<string>`, including `onChunk` behavior if the UI currently streams advice.

The agent must:

- Answer in Italian.
- Use the existing child profile and conversation context.
- Stay within the existing advice domains.
- Redirect current symptoms, diagnosis requests, and prescriptions to the existing triage or clinician workflow.
- Avoid fabricated medical claims, medication doses, and diagnoses.

## Phase 5: Implement the Casual Agent

The casual agent also returns text, but it has two existing modes:

```ts
type CasualMode = 'normal' | 'refusal';
```

Preserve both modes, the photo-upload guidance, doctor-office-hour injection, and the short-response policy. Keep the existing streaming mechanism because the frontend expects chunk callbacks.

```ts
export class CasualAgent {
  public constructor(
    private readonly cgptUtils: CGPTUtils,
    private readonly modelId: string
  ) {}

  public async respond(
    request: CasualAgentRequest,
    onChunk?: (chunk: string) => void
  ): Promise<string> {
    const messages = [
      { role: 'system', content: buildCasualSystemPrompt(request) },
      { role: 'user', content: buildCasualUserPrompt(request) },
    ];

    if (!onChunk) {
      const completion = await this.cgptUtils.retryAI(
        messages,
        this.modelId,
        undefined,
        undefined,
        undefined,
        DEFAULT_AI_TIMEOUT_MS,
        1024
      );
      return completion.choices[0]?.message?.content || '';
    }

    let response = '';
    for await (const chunk of this.cgptUtils.executeAIStream(
      messages,
      this.modelId,
      undefined,
      undefined,
      undefined,
      DEFAULT_AI_TIMEOUT_MS,
      1024
    )) {
      response += chunk;
      onChunk(chunk);
    }
    return response;
  }
}
```

`executeCasualModel()` should remain the public method and delegate to `CasualAgent.respond()`. Its current fallback user-facing error message remains unchanged.

## Phase 6: Wire the Three Agents

Add three fields to `AiRoutingService`:

```ts
private readonly triageAgent: TriageAgent;
private readonly generalChildCareAgent: GeneralChildCareAgent;
private readonly casualAgent: CasualAgent;
```

Initialize them with a shared `CGPTUtils` or `Gpt55AgentClient`. Resolve the existing `TRIAGE_MODEL_ID`, `ADVICE_MODEL_ID`, and `CASUAL_MODEL_ID` fields through `getAiCustomModelIDs()` so model changes remain runtime-configurable.

Replace method bodies only:

| Existing method | New delegation |
|---|---|
| `getUserQueryIntentAndSymptoms()` | `triageAgent.extract()` then `toTriageFlowIdentification()` |
| `executeAdvisorModel()` | `generalChildCareAgent.respond()` |
| `executeCasualModel()` | `casualAgent.respond()` |

Do not modify callers in `TypeBotService`. Its calls keep working because all three public method signatures remain stable.

## Tests and Acceptance Gates

Add focused Jest tests alongside the new modules. Mock `CGPTUtils`; tests must not call OpenAI.

### Triage unit tests

- Valid code is converted to its catalog label.
- Unknown code is rejected and never persisted.
- Model-provided label is ignored in favor of catalog lookup.
- Negated symptom is absent from `symptoms` and present only in `excluded` audit data.
- Past-resolved symptom is absent from `symptoms`.
- A hedged symptom is retained.
- Evidence span must occur in the input message.
- Generic illness with no extracted symptom returns `needsSymptomDescription: true`.
- Unmapped complaint is preserved in `newSymptoms`.
- Duplicate symptom codes result in one label.

### General child care tests

- Child context is included when provided.
- No child context does not produce malformed prompts.
- Symptom-like query returns the fixed triage redirect rather than medical advice.
- Tool response and direct text response both return a string.

### Casual tests

- Normal and refusal prompts use the correct mode.
- Office-hours data is injected only for an office-hours query.
- Streamed chunks are forwarded in order and joined into the return value.
- Stream failure uses the existing non-streaming fallback.

### Regression evaluation

Run the standalone suite before and after any prompt or catalog change:

```powershell
python cli.py evaluate --provider openai --model gpt-5.5-2026-04-23 --concurrency 10 --output data/eval/eval_gpt55_integrator_baseline.json
```

The promotion gate for the triage prompt is:

- Exact-message accuracy is at least 97.6 percent on the 860-case set, or a reviewed improvement is documented.
- No hallucinated code is accepted by adapter validation.
- Negation cases remain 100 percent correct.
- All new Jest tests pass.
- `npm run build --workspace=@medplum/server` passes.

## Rollout and Rollback

1. Deploy with all agents in `legacy` mode.
2. Enable triage in `shadow` mode for a representative sample. Log only hashed case IDs, selected codes, exclusion reasons, latency, model ID, and comparison outcome. Do not log raw pediatric messages.
3. Review disagreement cases with clinical product ownership.
4. Enable triage first. Keep advice and casual in `shadow` mode.
5. Enable general child care after prompt safety and profile-context tests pass.
6. Enable casual last after streaming and office-hours tests pass.
7. Roll back any agent immediately by setting its mode to `legacy`. No database migration is needed because public response contracts remain unchanged.

## Operational Metrics

Record these fields through the existing structured logger and Prometheus metrics:

| Metric | Tags or fields |
|---|---|
| Request latency | `agent`, `model`, `stream` |
| Model outcome | `agent`, `success`, `refusal`, `parse_failure` |
| Triage extraction | `confirmed_count`, `excluded_count`, `unmapped_count` |
| Adapter validation failure | `reason`, never raw message content |
| Shadow disagreement | `agent`, `legacy_result_hash`, `gpt55_result_hash` |
| Fallback activation | `agent`, `failure_type` |

Alert on increased parse failures, timeout rates, content-filter events, and unexpected empty triage output after router-selected triage messages.

## Implementation Order

1. Reconcile the catalog and generate the code-first `symptomCatalog.ts` file.
2. Add migration-mode configuration and a shared GPT-5.5 client.
3. Build and test `TriageAgent` plus the adapter.
4. Delegate `getUserQueryIntentAndSymptoms()` without changing any TypeBot code.
5. Run the 860-case GPT-5.5 evaluation and enable triage shadow mode.
6. Build and test the general child care agent while preserving `executeAdvisorModel()`.
7. Build and test the casual agent while preserving `executeCasualModel()` and streaming.
8. Promote each agent separately and retain one-command rollback to `legacy` mode.

## Definition of Done

- The three agent entry points retain their existing return types and callers.
- All three configured agent model IDs point to `gpt-5.5-2026-04-23`.
- Triage returns catalog-validated symptoms based on code, with evidence and exclusion handling.
- TypeBot receives the same `TriageFlowIdentification` shape without source changes.
- General child care and casual responses remain Italian, context-aware, and safety-constrained.
- The full GPT-5.5 triage evaluation, focused Jest tests, and Integrator server build pass.
- Each agent can be switched between `legacy`, `shadow`, and `enabled` without redeploying code.