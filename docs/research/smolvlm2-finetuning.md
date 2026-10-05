# SmolVLM2-500M-Video-Instruct — Fine-tuning Research Notes

Researched 2026-10-04. Sources: HF model card + raw repo configs, transformers docs (main / v5), TRL v1.14.1 docs + GitHub, PEFT PyPI, official HF fine-tuning notebooks (`huggingface/smollm/vision/finetuning/`).

## 1. Model card / architecture

Checkpoint: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (Apache-2.0, base = SmolVLM-500M-Instruct, architecture = Idefics3-family; SmolVLM2 = Idefics3 with SmolLM2 text model + multi-image/video support).

From `config.json` (verified raw):
- `model_type: "smolvlm"` (NOT `smolvlm2` — the 500M/256M checkpoints share the SmolVLM model type; `architectures: ["SmolVLMForConditionalGeneration"]`)
- **Vision encoder**: SigLIP-style (`model_type: "smolvlm_vision"`), hidden_size 768, 12 heads, image_size **512**, patch_size **16** (i.e. SigLIP-Base/512-class encoder, ~90M params; `use_base_siglip: false` is just a flag). Layers ≈ 12 (Idefics3/SigLIP default).
- **Connector**: Idefics3-style **pixel-shuffle** (pixel_shuffle_factor / scale_factor = **4**) — no resampler (`use_resampler: false`).
- **Language model**: `VLlama3ForCausalLM` = SmolLM2-360M-class: hidden_size 960, 32 layers, 15 heads, 5 KV heads, intermediate 2560, `max_position_embeddings: 8192`, vocab 49280. Total ≈ **500M params**.
- **Image resolution**: preprocessor `size: {"longest_edge": 2048}` (resize), `max_image_size: {"longest_edge": 512}` (splitting tile), `do_image_splitting: true`. Videos: `fps 1, max_frames 64, tile 512`.
- **Image tokens per image**: `processor_config.json` sets **`image_seq_len: 64`** (= (512/16)² / 4² per 512×512 tile). A small image (≤512px, 1 tile) expands to 64 (global) + 64 (tile) = **128 `<image>` tokens**; larger images add 64 tokens per extra 512px tile. The processor expands the single `<image>` placeholder automatically — never hardcode counts.
- **Dtype**: repo `torch_dtype: "float32"` on the top-level config; text_config `torch_dtype: "bfloat16"`. Model card loads with `torch.bfloat16` (CUDA). `use_cache: false` in the shipped config (correct for training).
- **Context length**: 8192 (`max_position_embeddings` and tokenizer `model_max_length: 8192`).

## 2. Config / tokenizer / chat template (verified raw files)

Special tokens (tokenizer_config.json, GPT2-BPE tokenizer, vocab 49152 + added → 49280):
- `<image>` (image_token_id **49190**), `<fake_token_around_image>`, `<end_of_utterance>`, `<global-img>` (extra_special_tokens)
- bos `<|im_start|>`, **eos `<end_of_utterance>`**, **pad `<|im_end|>`**; `truncation_side: "left"`

Chat template (chat_template.json == tokenizer_config, Jinja):
```
<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}
```
Rendered single-image example: `<|im_start|>User: <image>Is the invoice total above $500? Answer in JSON.<end_of_utterance>\nAssistant:{response}<end_of_utterance>`. Non-standard: capitalized roles, `:` glued to role when content[0] is an image, eos inside/after each turn, generation prompt is `Assistant:` with NO trailing space/newline.

### Messages format — one training example, one image
```python
from PIL import Image
img = Image.open("sample.png")  # pass PIL objects in the "images" column for TRL

example = {
    "images": [img],                      # TRL VLM SFT expects this column (list of PIL images)
    "messages": [
        {"role": "user", "content": [
            {"type": "image"},                             # placeholder; expanded by processor
            {"type": "text", "text": "Classify this screenshot. Reply with JSON."},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": '{"approved": true, "reason": "total < 500"}'},
        ]},
    ],
}
```
Plain-transformers equivalent: `processor.apply_chat_template(example["messages"], tokenize=True, return_dict=True, return_tensors="pt", images=example["images"])` (or put `{"type": "image", "image": img}` / `"url":` / `"path":` directly in content).

### Correct classes / calls
- `AutoProcessor.from_pretrained(model_id)` → `SmolVLMProcessor` (image proc + tokenizer + video proc).
- `AutoModelForImageTextToText.from_pretrained(model_id, ...)` → loads `SmolVLMForConditionalGeneration`. `AutoModelForVision2Seq` also maps it; direct `SmolVLMForConditionalGeneration` works. (Older notebooks used `Idefics3ForConditionalGeneration` for SmolVLM-Base; prefer AutoModelForImageTextToText.)
- Note: `torch_dtype=` was renamed to `dtype=` in transformers v5; `_attn_implementation` → `attn_implementation`.

## 3. Version history & requirements

- **transformers**: SmolVLM model support (model_type `smolvlm`) since ~v4.47.x (config written with 4.47.1). Dedicated **SmolVLM2** support (video, docs) merged via PR #36126 on 2025-02-20 → **first stable release with full SmolVLM2 support: v4.50.0** (v4.49.0, released 2025-02-17, predates the merge). In transformers **v5.x (current latest 5.18.0)** the separate `smolvlm2` model type was consolidated back into `models/smolvlm` — the 500M-Video checkpoint still loads fine because its `model_type` is `smolvlm`.
- **TRL**: latest stable **1.14.1** (requires `transformers>=4.56.2`, `datasets>=4.7.0`, `accelerate>=1.4.0`). `SFTTrainer` "fully supports" VLM training: pass a dataset with an `images` column (list of PIL) or `image` column (single PIL) + conversational `messages` whose `content` is a list of `{"type": "image"|"text", ...}` dicts. Reference dataset: `trl-lib/llava-instruct-mix`; format doc: "Dataset Formats → Vision datasets". Mixing text-only + vision rows requires transformers ≥ 4.57.0. **Set `SFTConfig(max_length=None)`** so truncation can't strip image tokens. Custom-collator path (official notebooks): `processor.apply_chat_template(..., tokenize=True, return_dict=True)`, pad with `processor.tokenizer.pad_token_id`, mask labels `-100` for pads and `image_token_id`.
- **TRL "vlm" extra**: `pip install trl[vlm]` pulls `Pillow`, `torchvision`, `num2words==0.5.14` (num2words is also required by the SmolVLM2 model card).
- **PEFT**: latest **0.21.2** (LoRA supports `exclude_modules` since 0.14, `use_dora`, `init_lora_weights="gaussian"`).

### LoRA guidance for this architecture (from official HF notebooks)
Official SmolVLM/SmolVLM2 notebooks (`huggingface/smollm/vision/finetuning/Smol_VLM_FT.ipynb`, `SmolVLM2_Video_FT.ipynb`) use:
```python
LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1,
           target_modules=["down_proj","o_proj","k_proj","q_proj","gate_proj","up_proj","v_proj"],
           use_dora=False, init_lora_weights="gaussian")
```
- **Attention + MLP projections of the LLM**: `q_proj, k_proj, v_proj, o_proj` (attention) + `gate_proj, up_proj, down_proj` (SwiGLU MLP). hidden 960 / intermediate 2560.
- **Gotcha — name collision**: the SigLIP vision tower also has `q_proj/k_proj/v_proj` (and fc1/fc2/out_proj MLP), so the bare-name list above attaches LoRA to the vision tower too. For a decision model where the visual task is fixed, freeze the vision tower: pass a regex like `target_modules=r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"` (verify exact prefix with `[n for n,_ in model.named_modules() if "proj" in n]` — transformers v5 splits into `model.vision_tower` / `model.connector` / `model.language_model`), or use PEFT `exclude_modules`. For full FT the notebook does `for p in model.model.vision_model.parameters(): p.requires_grad = False`.
- **Rank/alpha**: r=8, alpha=8, dropout=0.1 is the HF-endorsed baseline for ≤2.2B SmolVLMs; r=16/alpha=16 is a reasonable upgrade for structured-JSON behavior shaping. The HF blog explicitly recommends **full fine-tuning (not LoRA) for the 500M** variant (it's small enough); LoRA is still fine for smoke tests and low-memory hosts. Reference LR in notebooks: 1e-4 (LoRA) / 2e-5-ish for full FT; batch 2 + grad-accum + gradient checkpointing.

## 4. Apple Silicon (MPS) feasibility

- **flash-attention-2 is CUDA-only** — never pass `_attn_implementation="flash_attention_2"` on darwin; use `attn_implementation="sdpa"` (MPS-supported) or `"eager"`.
- **bf16 on MPS**: partially supported (many ops fall back to CPU / error); models pretrained in bf16 behave differently when run in other precisions. **fp16 attention overflows on MPS** (documented community finding, Aug 2026 rebuild-the-paper article: fp32 avoids overflow entirely). **Practical choice: run smoke tests in fp32** (CPU or MPS), LayerNorm/RMSNorm stay in fp32.
- Known MPS limitations: not all ops implemented → occasional errors/fallbacks; `device_map="auto"` (accelerate) can misbehave — keep everything on one device explicitly.
- **CPU fp32 smoke run is realistic**: 500M params × 4 bytes ≈ **2.0 GB** weights; a few steps on 1–2 short examples is minutes-scale. LoRA adds only ~2–5M trainable params → grads + AdamW states ≈ **~60–100 MB**; activations for seq ≈ 300–500 tokens (128 image tokens + text) are small. Total **~2.5–4 GB RAM**. Full fp32 fine-tune (all 500M trainable): weights 2 GB + grads 2 GB + AdamW 8 GB ≈ **12 GB** — fits a 16–32 GB Mac; MPS fp32 forward/backward works (slow, maybe 2–5 s/step at this size).
- CUDA runs: bf16 + flash_attention_2 per model card; 500M is tiny (model card: 1.8 GB GPU RAM for inference).

## 5. Pinned dependency set (verified latest stable, 2026-10-04)

```
torch==2.14.1            # MPS fp32 OK; CUDA builds for real runs
torchvision==0.29.1(?)   # match torch 2.14.1; required by TRL vlm extra
transformers==5.18.0     # >=4.56.2 needed by TRL 1.14.1; smolvlm model_type present in v5
trl==1.14.1              # SFTTrainer native VLM support (images column + messages)
peft==0.21.2             # LoraConfig, exclude_modules, use_dora
accelerate==1.15.0
datasets==5.0.1
pillow>=11               # PIL images for the images column
num2words==0.5.14        # required by SmolVLM2 processor/model card
# optional: decord (video input); NOT flash-attn on macOS
```
If TRL 1.14.1 + transformers 5.x integration friction appears, fall back to the last 4.x line (≥4.57) — TRL 1.14.1 accepts it.

## 6. Top gotchas (summary)

1. `model_type` is `smolvlm` (not `smolvlm2`); load via `AutoModelForImageTextToText` / `SmolVLMForConditionalGeneration`. The `smolvlm2` type existed only ~4.50–4.5x.
2. Non-standard chat template (capitalized roles, `Assistant:` no trailing space, `<end_of_utterance>` eos, `<|im_start|>` prefix) — always go through `processor.apply_chat_template`, never hand-format strings.
3. `<image>` is ONE placeholder token that the processor expands to `image_seq_len=64` tokens per tile (+global); truncation removing image tokens crashes training → TRL `SFTConfig(max_length=None)`; tokenizer `truncation_side` is "left".
4. LoRA target-module names collide with the SigLIP vision tower (`q_proj`,`k_proj`,`v_proj`); exclude the vision tower (regex / `exclude_modules`) or you'll also adapt vision weights.
5. MPS: no flash-attention, fp16 attention overflows, bf16 patchy → smoke-test in **fp32 on CPU/MPS** with `attn_implementation="sdpa"`/`"eager"`, `use_cache=False` during training; keep the whole model on one device.

## 7. Corrections verified at runtime (2026-10-04, first smoke train on this machine)

These override statements above where they conflict — found while running the actual LoRA smoke train (transformers 5.18.0, TRL 1.14.1, torch 2.14.1, MPS fp32):

1. **Image tokens are 1088 for ANY image size** (even 96×96): the stock processor performs image splitting unconditionally into 16 patch tiles + 1 global = 17 × `image_seq_len` 64 = 1088 `<image>` tokens. The "128 tokens for a ≤512px image" claim in §1/§6.3 holds only with `do_image_splitting: false`. Budget example lengths accordingly (~1600–1700 tokens with prompt + answers); our configs set `max_token_len: 2600` for image rows.
2. **Language tower module prefix is `model.text_model`** in transformers 5.18.0 (not `model.language_model`); vision = `model.vision_model`, bridge = `model.connector`. A safe LoRA regex covering both spellings: `model\.(text_model|language_model)\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)` — verified 224 language hits / 0 vision hits on the real model.
3. **transformers v5 TrainingArguments**: `warmup_ratio` was removed (convert to `warmup_steps`); `num_train_epochs` must be numeric (not None).
4. **MPS fp32 + sdpa works fine** for LoRA at this size: ~0.8 steps/s with 64-example batch-1 smoke (3 steps, 3.7 s total), loss 1.11 → 0.94, grad-norm ~0.46, token-accuracy ~0.75.
5. **TRL 1.14.1 conversational prompt-completion + `images` column** gives assistant-only masking out of the box (`completion_only_loss` auto-enabled) even though the SmolVLM2 chat template has no `{% generation %}` markers — no custom collator needed. Verified supervised tokens are exactly `<assistant JSON><end_of_utterance>`.
6. `apply_chat_template(..., add_generation_prompt=True)` appends exactly `Assistant:` (tokens `Ass`,`istant`,`:`) with no trailing space; for inference continuations, pass PIL images as `{"type": "image", "image": pil}` content parts (passing `images=` as well raises "got multiple values for 'images'" in v5).
