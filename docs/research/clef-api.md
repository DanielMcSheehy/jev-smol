# CLEF / Jev "System One" API — Research Notes (2026-10-04)

Purpose: exact API contract for cloning Cloudflare's CLEF decision model as "jev".
Everything below is quoted from primary sources (Cloudflare blog, Workers AI docs JSON
schemas, HF model cards, and the shipped inference code `joint_schema_model.py`).

## 1. What CLEF is

- CLEF and CLEF-flash are Cloudflare's open-weights (Apache-2.0) **decision models**:
  bounded, structured, probabilistic classifications for agentic workflows. They read a
  `state` (text, JSON, images, or video) plus a schema of typed questions and return a
  **probability for every allowed option of every question in a single forward pass** —
  no free-form generation, no output parsing.
- Announced: https://blog.cloudflare.com/clef-decision-models/ (released ~2026-09-30,
  HF repos last modified 2026-10-01). Hosted on Workers AI and open-sourced on HF.
- Positioning vs LLMs: typed answers with probabilities, deterministically; vs static
  classifiers: generalizes to new categories without retraining.
- **The API is not Cloudflare's invention.** Both model cards say: "The Clef API is fully
  compatible with Jev and SystemOne." The wire format is the **Jev / SystemOne API**
  (`POST /v1/systemone`) from TypeSafe AI's closed hosted model "Jev" (System One),
  released 2026-09-15. CLEF is Cloudflare's open multimodal implementation of that API.
- Cloudflare's internal use case: classifying website domains via Browser Run
  (e.g. "95% fashion, 85% ecommerce, <1% phishing"; 2.2s fetch/render/classify vs 4.7s
  for gpt-oss-120b).

## 2. Models, sizes, architecture

| | Clef | Clef-flash | (TypeSafe) Jev 1.13 | autotrust JEV-27B (open student) |
|---|---|---|---|---|
| Backbone | Qwen/Qwen3.8-27B (27.36B params, BF16) + vision encoder | Qwen/Qwen3.5-9B (9.41B params) | unpublished (closed) | Qwen3.8-27B text tower |
| Pipeline tag | image-text-to-text | image-text-to-text | — | text-classification |
| Context | 64k (65,536 on Workers AI) | 64k | ~32k (state "up to 32k tokens") | 262,144 native |
| Output | 1 logit per allowed option, softmax per question | same | distributions rounded to 2 decimals | 24-slot fp32 decision head |
| Vision | yes (PNG/JPEG/WebP) | yes | **no — text-only** | JEV-27B-VL variant only |
| License | Apache-2.0 | Apache-2.0 | closed | Apache-2.0 |

Architecture (from model cards + code): frozen Qwen backbone + a **joint schema head**
(a small transformer: per-choice evidence routing layers, cross-field decoder layers,
schema-bound scoring, lexical prior). Prefill-only, non-autoregressive. Clef head config
(`joint_head_config.json`): `hidden_size 5120, width 1024, routing_layers 2, layers 4,
heads 16, feedforward 4096`.

Training (blog): label-smoothed cross-entropy + Brier loss for calibration;
**RLCD** ("Reinforcement Learning for Calibrated Decisions") for partial credit on
adjacent ordinals, precision rewards, reference penalty; rank-256 low-rank adapters on a
frozen backbone; internal synthetic datasets. Built on earlier DiffusionGemma
(logprob-deterministic) work, inspired by Matt Mastracci's research and vLLM PRs.

## 3. THE API (the thing to clone)

Two names for the same wire format: **"System One" API** (TypeSafe) = **"Jev API"**.
Cloudflare serves it at `POST /v1/accounts/{id}/ai/run/@cf/cloudflare/clef`; TypeSafe at
`POST https://api.typesafe.ai/v1/systemone`; the HF `systemone()` function takes/returns
"the same response body". Model ids: `"clef"`, `"clef-flash"` (Cloudflare);
`"jev-latest"` / `"jev-1.13.0"` (TypeSafe).

### 3.1 Endpoint & auth

```
POST https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef
Authorization: Bearer $CLOUDFLARE_AUTH_TOKEN
Content-Type: application/json

# TypeSafe equivalent:
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
```
Workers binding: `env.AI.run("@cf/cloudflare/clef", {...})`.
Errors (TypeSafe, fail-closed): 401 invalid key, 422 validation (body names field),
429 rate limit, 529 overloaded. The HF reference implementation raises
`ValueError("model and state are required")`, `"at least one question is required"`,
`"{id}: type must be noul, choice, or score"`, `"{id}: criteria must not be empty"`.

### 3.2 Request body (verbatim Cloudflare input schema, from
https://developers.cloudflare.com/workers-ai/models/clef/schema-input.json)

```jsonc
{
  "model": "clef",              // required, pattern ^\s*(clef|clef-flash)\s*$
  "state": <string|object|array>, // required. Text or structured data. Truncated to fit token limit.
  "questions": {                // required. 1–64 questions; ids: letters/digits/_/./- , max 100 chars
    "<question_id>": {
      "type": "noul",           // "noul" | "choice" | "score"
      "instructions": "...",    // required, non-empty string (or object/array holding the question)
      "criteria": { "true": "...", "false": "..." }  // noul: OPTIONAL descriptions of yes/no
    },
    "<question_id>": {
      "type": "choice",
      "instructions": "...",
      "criteria": { "<option_id>": "<description>", ... }  // 2–255 options; desc may be string/object/array/null
    },
    "<question_id>": {
      "type": "score",
      "instructions": "...",
      "criteria": ["<level 0>", "<level 1>", ...]  // ordered array, lowest first, indexed from 0; 2–10 levels
    }
  },
  "images": [ ... ]             // OPTIONAL "Clef extension to the System One API".
                                // Max 4 images; PNG/JPEG/WebP; data URL string ("data:image/png;base64,...")
                                // or {"content_type":"image/png","base64":"..."}; 4 MiB & 16 Mpx each,
                                // 8 MiB total decoded, request body max 13 MiB. Remote URLs NOT accepted.
}
```
(Blog/Workers-AI docs also show a curl with `"model": "clef"`, and docs note `videos`
exists in the HF reference implementation even though Workers AI documents only
`images`.)

Verbatim curl example (blog + Workers AI docs):
```bash
curl https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef \
  -X POST -H "Authorization: Bearer $CLOUDFLARE_AUTH_TOKEN" -d '{
    "model": "clef",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
      "urgent":  { "type": "noul", "instructions": "Is this support request urgent?" },
      "team":    { "type": "choice", "instructions": "Which team should handle this request?",
                   "criteria": { "billing": "Payments, invoices, and refunds",
                                 "technical": "Outages, errors, and configuration",
                                 "sales": "Plans and upgrades" } },
      "severity":{ "type": "score", "instructions": "How severe is the customer impact?",
                   "criteria": ["No impact", "Minor", "Major", "Critical"] }
    }
  }'
```

Verbatim TypeSafe request examples (docs.typesafe.ai/api) — same shape, `"model": "jev-latest"`:
- noul with criteria: `{"state":"Help! My payouts have been failing for 3 days.","model":"jev-latest","questions":{"is_urgent":{"type":"noul","instructions":"Does this convey urgency?","criteria":{"true":"Explicitly time-sensitive","false":"No urgency expressed"}}}}`
- choice: `{"state":"Help! My payouts have been failing for 3 days.","model":"jev-latest","questions":{"department":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments, invoicing, refunds","technical":"Bugs, outages, integrations","sales":"Pricing, upgrades, new accounts"}}}}`
- score: `{"state":"Help! My payouts have been failing for 3 days.","model":"jev-latest","questions":{"frustration":{"type":"score","instructions":"How frustrated is the customer?","criteria":["Calm","Frustrated","Very angry"]}}}`

HF README example (record form, `systemone()`):
```python
response = systemone(model, processor, {
    "model": "clef",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle the message?",
                       "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
})
```
Notes: `instructions` is optional in the HF implementation (falls back to the question
id); it is *required* on Workers AI. Default max_length 16,384 tokens for
`encode_record`; state is truncated to fit.

### 3.3 Response body (verbatim Cloudflare output schema,
https://developers.cloudflare.com/workers-ai/models/clef/schema-output.json)

```jsonc
{
  "model": "clef",              // model that performed the evaluation (echoed/resolved)
  "answers": {                  // one answer per question, keyed by the request's question ids
    "<id>": {                   // noul answer:
      "type": "noul",
      "noul": 0.95              // number 0..1, "Probability the answer is yes"
    },
    "<id>": {                   // choice answer:
      "type": "choice",
      "choice": "billing",      // highest-probability option id
      "probabilities": { "<option_id>": 0.88, ... },  // sums to 1
      "confidence": 0.81        // 0..1, "derived from the probabilities"
    },
    "<id>": {                   // score answer:
      "type": "score",
      "score": 1.05,            // probability-weighted level; can land between levels
      "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
      "probabilities": { "0": 0.0, "1": 0.95, "2": 0.05 },
      "confidence": 0.92
    }
  },
  "usage": { "input_tokens": 296, "output_tokens": 20 }  // both integers; output_tokens 0 for CLEF
}
```

Verbatim response examples (docs.typesafe.ai/api), all `"model": "jev-1.13.0"`:
```json
{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.95}},"usage":{"input_tokens":296,"output_tokens":20}}
{"model":"jev-1.13.0","answers":{"department":{"type":"choice","choice":"billing","probabilities":{"billing":0.88,"technical":0.12,"sales":0.0},"confidence":0.81}},"usage":{"input_tokens":318,"output_tokens":34}}
{"model":"jev-1.13.0","answers":{"frustration":{"type":"score","score":1.05,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.0,"1":0.95,"2":0.05},"confidence":0.92}},"usage":{"input_tokens":304,"output_tokens":18}}
```

Exact answer construction from the reference code (`systemone_answer`,
probabilities rounded to 4 decimals; TypeSafe rounds to 2):
- noul: `{"type":"noul","noul": round(p["true"],4)}` (p = softmax over [false,true])
- choice: `choice = argmax`; `confidence = p[choice]`; `probabilities` over all option ids
- score: levels are `"0".."N-1"`; `score = Σ i·p_i` (expected value, can be fractional);
  `confidence = max p`; `legend = {level: description}`; `probabilities` keyed by level index
- usage: `{"input_tokens": len(encoded.input_ids), "output_tokens": 0}`

### 3.4 The exact prompt template (from `joint_schema_model.py`, shipped in the HF repo)

System prompt (verbatim constant):
```
Read the complete state and schema. Decide every field jointly. Each answer
must be exactly one of that field's allowed options.
```

Full serialized record (Qwen chat template, verbatim pieces):
```
<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
STATE:
{state rendered as JSON: json.dumps(state, ensure_ascii=False, separators=(",",":"), sort_keys=True); strings pass through raw}
[{media: "<|vision_start|><|image_pad|><|vision_end|>" per image, "<|vision_start|><|video_pad|><|vision_end|>" per video, then "\n" — placed right after "STATE:\n", i.e. before the state text}]

SCHEMA FIELDS:
FIELD 1
ID: {question_id}
TYPE: {noul|choice|score}
INSTRUCTION: {instructions, or the question id if omitted}
ALLOWED OPTIONS:
OPTION 1: {"option_id": "...", "description": "..."}
OPTION 2: {"option_id": "...", "description": "..."}
END FIELD
FIELD 2
...
END FIELD
<|im_end|>
<|im_start|>assistant
<think>

</think>

JOINT SCHEMA DECISIONS:
```
Rules from code:
- `choice` options are emitted **sorted by option id** (`sorted(criteria.items())`).
- `score` options are `OPTION i+1: {"option_id": str(index), "description": criteria[index]}`.
- `noul` options are fixed: `{"option_id":"true","description":"The proposition is true or the answer is yes."}`
  and `{"option_id":"false","description":"The proposition is false or the answer is no."}`,
  overridable by `criteria.true`/`criteria.false`.
- Option descriptions that are None are omitted from the JSON (just `{"option_id": ...}`).
- The head reads mean hidden states over each INSTRUCTION span and each OPTION span and
  scores all options of all questions jointly; one logit per option; softmax per question.
- Images are placed BEFORE the state text (media tokens prepended after "STATE:\n").

### 3.5 TypeSafe Jev's own prompt (different model, same API)
The open autotrust JEV-27B student uses a per-question template (one question per call):
```
[kind] {kind}
[state] {state}
[question] {question}
[options]
A) option1
B) option2
...
[decision]:
```
(`noul` options are `false/true`, `score` is `"0".."5"`; choice labels A–P native, up to
256 options with Q–Z, AA, AB…; 24-slot fp32 head, per-kind temperatures
noul 1.014 / choice 1.016 / score 1.004.)

## 4. Usage snippets

### transformers (from HF model card, verbatim)
```python
path = snapshot_download("Cloudflare/clef")
sys.path.insert(0, path)
from joint_schema_model import collate_records, encode_record, load_release_model
model, processor = load_release_model(path, device="cuda")
record = {"state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
          "questions": {"status": {"type": "choice", "instructions": "What is the invoice status?",
                                   "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
                        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}}}
encoded = encode_record(processor.tokenizer, record, processor=processor)
batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
with torch.inference_mode():
    logits = model(batch)[0]
for question, question_logits in zip(encoded.questions, logits):
    probabilities = question_logits.float().softmax(-1).tolist()
    print(question.question_id, dict(zip(question.option_ids, probabilities)))
```
Tested with torch 2.11, transformers 5.10.2, single H200. Images: add
`"images": [PIL.Image]` to the record + pass processor; text-only and multimodal records
can share a batch.

### Workers AI (TypeScript binding, verbatim from docs)
```ts
const response = await env.AI.run("@cf/cloudflare/clef", {
  model: "clef",
  state: "Checkout has been failing for every customer for the last hour.",
  questions: { /* same shape as REST example above */ },
});
return Response.json(response);
```
Pricing: $0.24 per M input tokens.

### vLLM
Not for CLEF (custom head; blog mentions vLLM PRs that made prefill-only scoring
possible). The autotrust JEV-27B card shows the vLLM pattern for Jev-style models:
OpenAI-compatible server + `POST /v1/decide` (`{kind, state, question, options}` →
`{probabilities, choice, choice_index, usage, ...}`), or `/v1/completions` with
`max_tokens=1`, `allowed_token_ids` = option verbalizer tokens, then head bias +
per-kind temperature applied client-side.

## 5. Evals (Decision Index 0.2.1, from HF card; %, best in bold)

| Benchmark | Clef | Clef-flash | Jev | DiffusionGemma Jev | Kev 9B | Laya |
|---|---|---|---|---|---|---|
| BFCL (case exact) | 98.5 | **98.8** | 95.8 | 96.5 | 94.5 | 38.1 |
| ToolRet nDCG@10 | **69.2** | 66.4 | 65.3 | 61.2 | 64.3 | 12.8 |
| API-Bank | 91.9 | **93.1** | 88.2 | 83.7 | 56.3 | 11.5 |
| BANKING77 macro-F1 | **94.2** | 90.9 | 79.7 | 74.3 | 84.8 | 14.3 |
| CLINC150+OOS | **97.4** | 66.8 | 89.3 | 83.5 | 79.0 | 3.2 |
| MMLU | 90.3 | **91.8** | 91.7 | 79.3 | 75.3 | 30.7 |
| GSM8K | **80.8** | 67.3 | 79.9 | 50.3 | 48.7 | 21.6 |
| PhishNChips | 79.6 | 75.0 | 62.5 | **85.4** | 50.7 | 50.1 |
| Median latency (ms) | 209.3 | 38.8 | 524.1 | 84.4 | 51.4 | **5.8** |
| p95 latency (ms) | 238.6 | **122.4** | 536.0 | 211.2 | 187.9 | 222.5 |

Workflow evals (Typesafe Evals, evals.typesafe.ai): invoice processing exact actions
Clef 64.7 / flash 57.1 / Jev 61.8; customer service 76.3/77.0/76.0; security incidents
62.9/61.7/61.7; agent trace observability 68.5/69.8/**71.6** (Jev best).
Live demo: https://clef-evals.workers-ai-mle.workers.dev

## 6. Training data / code released?

- **Cloudflare: no public training data or training code.** HF `api/datasets?author=Cloudflare`
  returns `[]`; no github.com/cloudflare clef repo (GitHub search: zero cloudflare-org hits).
  Blog announces an RL fine-tuning platform (AI Gateway traffic capture → Workers AI
  rollouts → Containers RL sandbox → "Trainer" → redeploy via Workers AI BYO Model/Cog)
  with an interest form (cloudflare.com/resource/clef-rl-interest); training is
  "hands-on with the FDE team", self-serve planned — i.e. infra only, no released data.
- **The Jev ecosystem is the opposite: everything is public.**
  - Distillation corpus: `SargeDev/jev-distill-corpus-v3` (740,957 rows; `yuri_v3`
    498,010 rows of TypeSafe Jev 1.13 full output distributions via OpenRouter,
    `openjev_v2` 94,801 rows programmatic labels CC0, `yuri_v1` placeholders).
  - Datasets: `tasksource/tasksource-jev-typed-decisions`, `samatv256/jev-decisions-v1`,
    `Praveenrajus/jev-bench`, `FINAL-Bench/Darwin-27B-JEV-decision-index`, etc.
  - Open reproductions: `autotrust/JEV-9B` + `JEV-27B` (+`-VL`) "Blocks of Experts"
    (LoRA r=16 + 24-slot decision head distilled from Jev 1.13, KL≈0.017);
    `ZefanCai/Open-Jev-2B/9B/27B-v1.1`; `denis-pplx/autojev-27b`;
    `TokenRhythm/NeoHorse-Jev-4B`; `chaoliangUNSW/Jev-Style-*-GGUF`;
    `autotrust/JEV-Gemma4-26B-A4B`; community tracker
    https://huggingface.co/spaces/multimodalart/jev-decision-index (55 reproductions).
  - DiffusionGemma angle: Google's DiffusionGemma (26B MoE, 3.8B active, Gemma 4 base,
    256-token block denoising) has a "Jev-style output" variant ("DiffusionGemma Jev",
    community project "djev"/Kev — "parallel decisions with DiffusionGemma"); it scores
    85.4 on PhishNChips (best in the Decision Index table) but loses to Clef elsewhere.

## 7. What "jev" is (name check)

**Jev is a real, existing, closed hosted model/API — not just the user's project name.**
TypeSafe AI's "Jev" (System One), released 2026-09-15: a text-only, decision-only model
("Jev decides, code controls, the LLM generates when generation is needed"), ~32k-token
state, 70–500 ms end-to-end, model id `jev-latest` / `jev-1.13.0`, API at
`POST api.typesafe.ai/v1/systemone`. Its output primitives (noul/choice/score) and API
are the industry wire format — Cloudflare explicitly made CLEF "Jev-API compatible".
The user's clone being named "jev" and wanting "the api as jev" means: implement the
Jev/SystemOne API (which is exactly the CLEF contract documented above, plus CLEF's
`images` extension for the text+image requirement).

## 8. Source links

- Blog: https://blog.cloudflare.com/clef-decision-models/
- Workers AI docs: https://developers.cloudflare.com/workers-ai/models/clef/
  + https://developers.cloudflare.com/workers-ai/models/clef/schema-input.json
  + https://developers.cloudflare.com/workers-ai/models/clef/schema-output.json
- Model cards: https://huggingface.co/Cloudflare/clef ,
  https://huggingface.co/Cloudflare/clef-flash (READMEs + `joint_schema_model.py`,
  `joint_head_config.json`, `generation_config.json` fetched from raw HF)
- TypeSafe Jev: https://docs.typesafe.ai/api , https://jevtypesafeai.com ,
  https://jev.pro , https://systemonemodels.tech
- Open students / ecosystem: https://huggingface.co/autotrust/JEV-27B ,
  https://huggingface.co/autotrust/JEV-9B , https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1 ,
  https://huggingface.co/SargeDev/jev-distill-corpus-v3 ,
  https://huggingface.co/spaces/multimodalart/jev-decision-index ,
  https://github.com/gazelle93/decision-models-under-pressure ,
  https://evals.typesafe.ai/ , https://clef-evals.workers-ai-mle.workers.dev
- GitHub community: lucataco/clef-webcam, HighlyLoadedEgo/ClefMCP, Gjusev/clef-router,
  MersivMedia/clef-finetune, JordanDalton/clef-playground, Gjusev/clef-evals,
  hanyeol/mindor-clef-trainer, tehtommeh/clef-demo
