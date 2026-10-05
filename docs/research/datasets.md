# Dataset research for jev training data (text + image decision examples)

Researched 2026-10-04. Goal: public, license-clear HF datasets to build ~5k-50k
"decision-style" training examples (prompt + optional image -> structured JSON decision
with rationale), mimicking Cloudflare's CLEF decision models.

## 1. Cloudflare CLEF training data: NOT FOUND

- Cloudflare's HF org (`huggingface.co/Cloudflare`) has released **models, not datasets**:
  `Cloudflare/clef` (27B, Apache-2.0, post-trained from `Qwen/Qwen3.8-27B`) and
  `Cloudflare/clef-flash` (9B). Model card: <https://huggingface.co/Cloudflare/clef>,
  blog: <https://blog.cloudflare.com/clef-decision-models>.
- `https://huggingface.co/api/datasets?author=Cloudflare` returns **[]** (empty).
  Searches for "cloudflare" / "CLEF" datasets surface only unrelated sets
  (CLEF conference check-worthiness corpora, cleft-palate audio, etc.).
- Web sources indicate training data was not released — only open weights plus a
  hosted RL-fine-tuning service. Nothing to reuse directly.
- Useful detail from the card: "The Clef API is fully compatible with Jev and
  SystemOne", and output is one logit per option per question (typed answers +
  probabilities, no free-form generation). Our training examples should mirror that
  shape: `state` (+ schema of typed questions) -> option choice + rationale.

## 2. Comparison table

### Text decision-flavored

| Dataset id | License | ~Size | Columns (key) | Notes |
|---|---|---|---|---|
| `openbmb/UltraFeedback` | MIT | 63,967 rows | `instruction`, `completions[].annotations.<aspect>.{Rating, Rationale}` , `completions[].response` | GPT-4 judgments with **written rationales** (helpfulness/honesty/truthfulness/instruction-following). Best judgment-with-reasoning source. |
| `stanfordnlp/snli` | CC BY-SA 4.0 | 550k train | `premise`, `hypothesis`, `label` (entail/neutral/contradict; -1 = no gold) | Clean 3-way decision task; no explanations, but trivially templated. |
| `batalovme/esnli_with_rationale` | (card blank; e-SNLI builds on SNLI, CC BY-SA 4.0 lineage) | ~550k train | `premise`, `hypothesis`, `label`, `rationale` | Human NLI **explanations**. Use small slice. |
| `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` | CC BY 4.0 | ~11k train / 4.4k test | `text`, `text_type`, `labels_0..labels_4` (annotators; "safe" or category) | Guardrail/safety classification, permissive. |
| `codelion/optillm-router-dataset` | Apache 2.0 | ~1-10k | `prompt`, `results[].{approach, rank, tokens, content}` | LLM-technique routing: pick the best-ranked approach -> routing decision. |
| `withmartian/routerbench` | **none stated on card** | ~30k rows (pkl files, 100MB-1.2GB) | prompt, per-model cost/correct score | Good routing signal but no license and only `.pkl` (not streaming). Optional; prefer optillm set. |
| `lmsys/toxic-chat` | CC BY-NC 4.0 | ~10k | user_input, toxicity/jailbreak labels | **Non-commercial — skip** for an embeddable model. |
| `facebook/anli` | CC BY-NC 4.0 | 160k+ | premise/hypothesis/label/reason | **Non-commercial — skip**. |

### Image + text

| Dataset id | License | ~Size | Columns (key) | Notes |
|---|---|---|---|---|
| `Multimodal-Fatima/VQAv2_sample_train` | CC BY 4.0 (VQAv2 terms; card inherited) | 1,000 rows | `image`, `question`, `multiple_choice_answer`, `answers`, `blip_caption` | Tiny, immediately usable VQA slice. (`Multimodal-Fatima/VQAv2_train` is the full train split if more is needed.) |
| `lmms-lab/textvqa` (redirects to `lmms-lab-encoder/textvqa`) | none on card; images are OpenImages **CC BY 2.0** | ~29k train | `image`, `question`, `answers` (list), `ocr_tokens`, `image_classes` | Scene-text VQA; `ocr_tokens` enable "grounded in the image" rationales. |
| `lmms-lab/DocVQA` (-> `lmms-lab-encoder/DocVQA`), config `DocVQA` | card says Apache-2.0 (upstream DocVQA commonly treated as CC BY 4.0 — verify before commercial use) | ~39k validation+test rows kept | `image`, `question`, `answers` (list), `question_types` | Document QA; card license is unusually permissive for a DocVQA mirror — re-verify. |
| `lmms-lab/ai2d` (-> `lmms-lab-encoder/ai2d`) | **none on card**; original AI2D is research-oriented — verify | ~3k test | `image`, `question`, `options` (4), `answer` | Diagram multiple-choice; MC format maps 1:1 to CLEF-style typed questions. |
| `jxie/coco_captions` | none on card; COCO images CC BY (Flickr TOS) | ~590k (karpathy split) | `image`, `caption`, `cocoid` | Use small slice for image-description/"identify" style decisions. |
| `HuggingFaceM4/ChartQA` | **GPL-3.0 on card** (upstream research) | ~30k | image/chart Q/A | Copyleft — avoid for embeddable training data. |
| `HuggingFaceM4/DocumentVQA` | none on card; script-based (viewer unsupported) | — | — | Prefer `lmms-lab/DocVQA` above. |

License hygiene: prefer MIT / Apache-2.0 / CC BY / CC BY-SA. Explicitly avoided:
CC BY-NC (`lmsys/toxic-chat`, `facebook/anli`), GPL (`ChartQA`), no-license-unclear
(`withmartian/routerbench`, `ai2d` — use caution or verify).

## 3. Recommended starter mix (~25k total, ~balanced text/image)

| # | Dataset id | License | ~Count | Why it fits / transform idea |
|---|---|---|---|---|
| T1 | `openbmb/UltraFeedback` | MIT | 5,000 | Judgment verdicts: ask "which response is better on truthfulness?" -> JSON `{verdict, score, rationale}` using the GPT-4 Rationale field verbatim. |
| T2 | `stanfordnlp/snli` | CC BY-SA 4.0 | 3,000 | 3-way NLI -> typed question "Does the premise entail the hypothesis?" -> `{answer: entailment/neutral/contradiction, rationale}` (template rationale from premise/hypothesis spans). |
| T3 | `batalovme/esnli_with_rationale` | CC BY-SA 4.0 lineage | 2,500 | Same NLI decision but the human `rationale` text becomes the natural-language explanation field. |
| T4 | `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` | CC BY 4.0 | 1,500 | Guardrail decision: `{safe: bool, category, rationale}` from majority `labels_*`. |
| T5 | `codelion/optillm-router-dataset` | Apache 2.0 | 500 | Routing decision: given prompt + per-approach results, `{selected_approach, why}` from rank/tokens. |
| — | **Text subtotal** | | **12,500** | |
| I1 | `Multimodal-Fatima/VQAv2_sample_train` | CC BY 4.0 | 4,000 | VQA -> grounded decision about the image: `{answer, rationale}` citing the question and the ten-annotator consensus (`answers` list); MC "is the answer X?" yes/no variants too. |
| I2 | `lmms-lab/textvqa` | CC BY 2.0 images (card blank) | 3,000 | Scene-text reading decision: answer about text visible in image; rationale may cite `ocr_tokens` actually present (drop examples where gold answer ∉ ocr_tokens). |
| I3 | `lmms-lab/DocVQA` (config `DocVQA`, validation) | Apache-2.0 per card (verify) | 2,500 | Document-understanding decision: `{answer, question_type, rationale}`; use `question_types` as the typed-question schema (extraction/reasoning/counting). |
| I4 | `lmms-lab/ai2d` | verify (research) | 2,000 | Native multiple-choice: `{selected_option, rationale}` — closest match to CLEF's "one logit per option" output. |
| I5 | `jxie/coco_captions` | COCO CC BY (images) | 1,000 | Description/verification decisions: show image + candidate caption -> `{true: bool, rationale}`; or caption -> "what is the main object?" from `caption`. |
| — | **Image subtotal** | | **12,500** | |
| | **Total** | | **25,000** | Within the 5k-50k target; scale any row up (e.g. SNLI to 10k, TextVQA to 8k) for a 50k set. |

## 4. Loading snippets

```python
from datasets import load_dataset

# T1 UltraFeedback (MIT) — instruction + per-aspect judgments with Rationale
ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
#   cols: instruction, completions[].response, completions[].annotations.<aspect>.{Rating,Rationale}

# T2 SNLI (CC BY-SA 4.0)
ds = load_dataset("stanfordnlp/snli", "plain_text", split="train", streaming=True)
#   cols: premise, hypothesis, label (0 entail,1 neutral,2 contradict, -1 no gold)

# T3 e-SNLI with human rationales
ds = load_dataset("batalovme/esnli_with_rationale", split="train", streaming=True)
#   cols: premise, hypothesis, label, rationale

# T4 AEGIS safety classification (CC BY 4.0)
ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split="train", streaming=True)
#   cols: text, text_type, labels_0..labels_4 ("safe" or harm category)

# T5 optillm router (Apache 2.0)
ds = load_dataset("codelion/optillm-router-dataset", split="train", streaming=True)
#   cols: prompt, results[].{approach, rank, tokens, content}

# I1 VQAv2 sample (CC BY 4.0) — 1,000 rows; plain load is fine
ds = load_dataset("Multimodal-Fatima/VQAv2_sample_train", split="train")  # ~small download
#   cols: image (Image), question, multiple_choice_answer, answers (10 annotator answers)

# I2 TextVQA (OpenImages CC BY 2.0)
ds = load_dataset("lmms-lab/textvqa", split="train", streaming=True)
#   cols: image (Image), question, answers (list), ocr_tokens (list)

# I3 DocVQA (card: Apache-2.0; verify for commercial use)
ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
#   cols: image (Image), question, answers (list), question_types (list)

# I4 AI2D multiple-choice (license: verify)
ds = load_dataset("lmms-lab/ai2d", split="test", streaming=True)
#   cols: image (Image), question, options (list of 4), answer

# I5 COCO captions
ds = load_dataset("jxie/coco_captions", split="train", streaming=True)
#   cols: image (Image), caption, cocoid
```

Notes on snippets:
- `lmms-lab/textvqa`, `lmms-lab/ai2d`, `lmms-lab/DocVQA` redirect on the Hub to
  `lmms-lab-encoder/*`; the ids above still resolve (they are the same data). Use
  the `lmms-lab-encoder/...` id directly if the redirect errors in your `datasets`
  version.
- Streaming pulls each parquet shard lazily; cap with `ds.take(n)`.
- TextVQA `answers` are 10 crowdsourced strings (majority vote = gold).
- SNLI rows with `label == -1` (no consensus) should be dropped.

## 5. Sources

- Cloudflare clef model card: https://huggingface.co/Cloudflare/clef
- Cloudflare blog (clef decision models): https://blog.cloudflare.com/clef-decision-models
- HF datasets API: https://huggingface.co/api/datasets (license/size per dataset card)
- datasets-server (columns/splits/row counts): https://datasets-server.huggingface.co
