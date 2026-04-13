# Project Context

- **Owner:** Bruno Capuano
- **Project:** Qwen3-TTS → ONNX → C# .NET 10 console app for local voice generation
- **Stack:** Python (PyTorch, transformers, ONNX), C# (.NET 10, ONNX Runtime), Qwen3-TTS
- **Created:** 2026-02-21T15:38Z

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-02-21: Qwen3-TTS Architecture Deep Dive

**Pipeline:** Text → BPE → Text Embedding (2048d) → Projection (→1024d) → Talker LM (20-layer, GQA 16/2, M-RoPE) → Code Predictor (5-layer, 31 autogressive steps per Talker step) → 32-codebook codes → Vocoder (16 codebooks used, RVQ dequantize → 8-layer transformer → BigVGAN-style conv decoder, 1920× upsample → 24kHz PCM)

**Key architecture facts:**
- Talker LM: hidden_size=1024, vocab_size=3072, num_code_groups=32
- Code Predictor: 5 layers, hidden_size=1024, vocab_size=2048, predicts groups 1-31 sequentially
- Vocoder (Tokenizer-12Hz): Decoder only (no ONNX internally), 16 RVQ codebooks, codebook_size=2048, sliding_window=72
- 12Hz tokenizer is pure PyTorch — no ONNX files loaded at runtime
- 25Hz tokenizer (not our target) DOES use ONNX internally
- Total upsample factor = 1920× (12 Hz codes → 24kHz audio)

**ONNX export plan:** 3 models — Vocoder (easiest, single pass), Code Predictor (small, 31-step loop), Talker LM (largest, KV-cache).

**Key source files:**
- `qwen_tts/core/models/modeling_qwen3_tts.py` — Talker LM + Code Predictor + Speaker Encoder
- `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` — Vocoder
- `qwen_tts/inference/qwen3_tts_model.py` — Inference orchestration

**Environment:** `python/` directory created with requirements.txt, download_models.py, README.md, ARCHITECTURE.md

### 2026-02-21: Talker LM + Code Predictor ONNX Export Scripts

Created `python/export_lm.py` and `python/export_embeddings.py` for the two autoregressive components.

**ONNX export approach:**
- Wrapper `nn.Module` classes that take/return flat KV-cache tensors instead of HF DynamicCache
- DynamicCache is reconstructed inside the wrapper forward() from flat tensors, then extracted after model call
- This traces cleanly through `torch.onnx.export` because list operations (append/index) execute at trace time; only the tensor operations (concat, matmul) appear in the ONNX graph

**Talker LM split into two ONNX models:**
- `talker_prefill.onnx` — full sequence, produces initial KV cache (20 layers × K,V of shape B,2,T,128)
- `talker_decode.onnx` — single token step, consumes and produces updated KV cache
- Both use M-RoPE position_ids (3, B, T) as input; rotary embedding module baked into ONNX graph

**Code Predictor — single ONNX model:**
- `code_predictor.onnx` — handles both prefill (S=2) and decode (S=1) depending on input shape
- 31 lm_head weight matrices stacked into a single buffer (31, 2048, 1024), indexed via `torch.index_select` on generation_steps input — avoids ModuleList dynamic indexing in ONNX
- Uses standard 1D RoPE (not M-RoPE), 5 layers, 8 KV heads

**Embedding extraction (`export_embeddings.py`):**
- Saves text_embedding, text_projection MLP weights, talker codec_embedding (3072×1024), 31 code predictor codec embeddings (2048×1024 each), codec_head weights as .npy files
- Exports speaker_ids.json (maps speaker names → token IDs in talker codec_embedding) and config.json with all special token IDs and model dimensions
- C# loads these directly as raw tensors — no ONNX overhead for simple lookups

### 2026-02-21: Vocoder ONNX Export Scripts

Created `python/export_vocoder.py` and `python/validate_vocoder.py` for exporting the `Qwen3TTSTokenizerV2Decoder` to ONNX.

**Decoder forward method** (verified from source):
```python
def forward(self, codes):  # (B, 16, T) int64
    hidden = self.quantizer.decode(codes)      # RVQ dequantize → (B, codebook_dim, T)
    hidden = self.pre_conv(hidden).transpose(1, 2)  # → (B, T, 1024)
    hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state  # 8-layer transformer
    hidden = hidden.permute(0, 2, 1)           # → (B, 1024, T)
    for blocks in self.upsample:               # 2×, 2× → (B, 1024, T×4)
        for block in blocks: hidden = block(hidden)
    wav = hidden
    for block in self.decoder:                 # BigVGAN: 8×5×4×3 → (B, 1, T×1920)
        wav = block(wav)
    return wav.clamp(min=-1, max=1)
```

**Export strategy:** Two-attempt approach — trace-based first (opset 17), dynamo fallback (opset 18+). Dynamic axes on batch and time dimensions. The model is loaded from `Qwen/Qwen3-TTS-Tokenizer-12Hz` via `AutoModel.from_pretrained(..., trust_remote_code=True)` and the decoder extracted as `model.decoder`.

**Potential ONNX issues to watch for at runtime:**
- Sliding-window attention (window=72) in the 8-layer transformer may generate data-dependent masks
- Causal convolution padding (`_get_extra_padding_for_conv1d`) uses input-length-dependent logic
- RVQ decode has a for-loop over 16 codebook layers (should unroll since count is fixed)
- SnakeBeta activation (`x + (1/β) * sin²(αx)`) uses only standard math ops — should be fine

### 2026-02-21: BPE Tokenizer Extraction & Prompt Format Documentation

**Tokenizer type:** Qwen2Tokenizer (GPT-2 style BPE, byte-level fallback). Uses `vocab.json` (151,936 entries) + `merges.txt`.

**Chat template format (CustomVoice):**
- Text wrapped as: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
- Instruct wrapped as: `<|im_start|>user\n{instruct}<|im_end|>\n` (prepended as embedding, not used for 0.6B model)
- First 3 tokens (`<|im_start|>`, `assistant`, `\n`) are the "role prefix" — embedded separately

**Codec prefix structure (Talker LM embedding space, NOT text tokens):**
- Language="auto": `[nothink(2155), think_bos(2156), think_eos(2157)]`
- Language explicit: `[think(2154), think_bos(2156), lang_id, think_eos(2157)]`
- Then: `[speaker_embed?, pad(2148), bos(2149)]`

**Key config IDs:** im_start=151644, im_end=151645, assistant=77091, tts_bos=151672, tts_eos=151673, tts_pad=151671

**Dialect auto-mapping:** Eric → Sichuan (2062), Dylan → Beijing (2074) when language is "chinese" or "auto"

**0.6B limitation:** Instruct parameter is forced to None for tts_model_size="0b6"

**Files created:** `python/extract_tokenizer.py`, `python/TOKENIZER.md`, `python/tokenizer_artifacts/` (output dir)

### 2026-02-21: GPU Handoff
All Python export scripts are written and committed. None have been run yet — they need model weights downloaded first. On the GPU machine, run `download_models.py` first, then export scripts in order: vocoder → LM → embeddings → tokenizer. Watch for attribute path issues in export_lm.py — the model attribute discovery code has fallbacks but may need adjustment.

### 2026-02-22: Qwen3-TTS 1.7B Technical Viability Analysis

**Issue #26 analysis complete.** The 1.7B model is architecturally compatible with our 0.6B ONNX pipeline — only the Talker LM hidden size changes (1024→2048). Code Predictor (1024), vocoder, tokenizer, and speaker inventory remain identical.

**Key findings:**
- **Model variants:** 0.6B has Base+CustomVoice (no instruct). 1.7B has Base+CustomVoice+VoiceDesign (both with instruct). "Instruct" means natural-language style control (emotion, rate, timbre), not GPT-style instructions.
- **Architecture:** Only `talker_config.hidden_size` (1024→2048) and `intermediate_size` (3072→6144) differ. Everything else (28 layers, 16 heads, 8 KV heads, head_dim=128, 16 code groups) is identical.
- **Parameter count:** 0.6B = ~458M params (~1.8 GB FP32). 1.7B = ~1344M params (~5.4 GB FP32). Delta: +886M params (+3.5 GB).
- **ONNX export feasibility:** Fully compatible. `python/export_lm.py` hardcodes `TALKER_HIDDEN = 1024` at line 29 — needs to read `model.config.talker_config.hidden_size` dynamically. No fundamental blockers.
- **C# changes required:** `LanguageModel.cs` and `EmbeddingStore.cs` have ~100+ hardcoded `1024` references in buffer allocations. Solution: add `HiddenSize` property to config, replace all `1024` with `config.Talker.HiddenSize`.
- **Resource requirements:** 16 GB RAM minimum for 1.7B (vs 8 GB for 0.6B). Inference ~2-2.5× slower. KV cache size per token is identical (224 KB) since head structure doesn't change.
- **Implementation effort:** 8-12 hours for MVP (config-driven exporters + C# refactor + validation).

**Documentation:** Full analysis written to `.squad/decisions/inbox/trinity-17b-viability.md` (16 KB, 9 sections covering variants, instruct semantics, architecture diffs, ONNX export, resource requirements, code changes, testing strategy, risks, recommendations).

**Next steps:** Pending team decision. If approved, Phase 1 = make Python exporters config-driven → export 1.7B → refactor C# to be dimension-agnostic → add model size selection to TtsPipeline.

### 2026-04-02T16:43Z: 1.7B Viability Analysis Complete & Approved for Implementation

📌 Team update (2026-04-02T16:43Z): Orchestration logs and decision consolidation complete. Both Trinity and Morpheus recommend pursuing Phase 1 1.7B support (8-12 hours MVP). Non-breaking change, high user value (instruction control). Awaiting maintainer approval to start Phase 1 implementation. — Scribe

### 2026-04-02: Python Export Scripts Made Config-Driven for 1.7B Support

**Issue #26 Phase 1 (Python side) — complete.**

**What changed:**
- `export_lm.py`: Removed all hardcoded model constants (`TALKER_HIDDEN=1024`, `CP_HIDDEN=1024`, etc.). Added `read_model_dims(config)` function that extracts dimensions from `config.talker_config` and `code_predictor_config`. Wrapper classes (`TalkerPrefillWrapper`, `TalkerDecodeWrapper`, `CodePredictorWrapper`) now accept `num_layers` as constructor arg. Export functions accept `dims` dict.
- `reexport_lm_novmap.py`: Same config-driven refactor + added `argparse` CLI with `--model-dir` and `--output-dir` (previously hardcoded paths).
- `reexport_base_novmap.py`: Same config-driven refactor + added `argparse` CLI.
- `download_models.py`: Added 1.7B variants. New `--model` choices: `customvoice-1.7b`, `all-1.7b`, `base-1.7b`, `everything`.
- `extract_tokenizer.py`: Added `--model` CLI arg. Tokenizer is shared across variants.
- `export_embeddings.py`: Already config-driven; updated docstring to document 1.7B support.
- `python/README.md`: Added comprehensive 1.7B documentation including export commands, GPU requirements, and time estimates.
- `python/ARCHITECTURE.md`: Updated Talker LM config table to show 0.6B vs 1.7B comparison.

**Key pattern — `read_model_dims(config)`:**
```python
def read_model_dims(config):
    tc = config.talker_config
    cp = tc.code_predictor_config
    return {
        "talker_num_layers": tc.num_hidden_layers,
        "talker_hidden": tc.hidden_size,
        "cp_num_layers": cp.num_hidden_layers,
        "cp_hidden": cp.hidden_size,
        ...
    }
```
This pattern should be reused if any new export scripts are added.

**Unchanged files (no action needed):**
- `export_vocoder.py` — vocoder architecture is independent of Talker hidden size
- `export_speaker_encoder.py` — speaker encoder already reads config
- `validate_vocoder.py` — vocoder-only validation
- `patch_models_for_dml.py` — auto-detects dimensions from ONNX graph

**0.6B backward compatibility:** All scripts retain 0.6B as default. Running without args produces identical behavior to before.

📌 Team update (2026-04-02T1719): Phase 1 complete — multi-variant support (0.6B and 1.7B) implemented across C#, Python, and tests. Orchestration logs and decisions merged. Non-breaking change, 88 tests pass. — Scribe

### 2026-04-02: Full 1.7B ONNX Export Pipeline — Complete

**Environment:** NVIDIA A10-24Q (24 GB VRAM), Python 3.13.9, PyTorch 2.6.0+cu124, transformers 4.57.3, onnx 1.20.1

**Model:** Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (3.57 GB safetensors, BF16)

**Exported artifacts (all to `python/onnx_1.7b/`):**

| File | Size | Notes |
|------|------|-------|
| `talker_prefill.onnx` + `.data` | 2.7 MB + 5.27 GB | 28 layers, hidden=2048 |
| `talker_decode.onnx` + `.data` | 2.7 MB + 5.27 GB | Same weights, decode-step graph |
| `code_predictor.onnx` + `.data` | 0.2 MB + 428 MB | 5 layers, hidden=1024 (unchanged from 0.6B) |
| `vocoder.onnx` | 436 MB | Shared across 0.6B/1.7B, validated (max err 2.5e-5) |
| `embeddings/*.npy` + config | ~1.5 GB total | 15 CP embeddings (2048×2048), text_embedding (151936×2048) |
| `tokenizer/` | ~4 MB | Shared vocab.json + merges.txt |

**Total export size: ~12.8 GB** (vs ~5.5 GB for 0.6B — roughly 2.3× larger)

**Issues encountered and fixed:**

1. **vmap masking incompatibility (transformers 4.57.3):** `torch.onnx.export` trace fails with `RuntimeError: invalid unordered_map<K, T> key` because transformers' `masking_utils.py` uses `torch.vmap` for causal mask creation, which is incompatible with JIT tracing. **Fix:** Use `reexport_lm_novmap.py` which registers `sdpa_without_vmap` from `transformers.integrations.executorch`. Applied same fix to `export_vocoder.py`.

2. **Code Predictor dummy input dimension bug (1.7B-only):** `export_lm.py` and `reexport_lm_novmap.py` created dummy CP inputs with `dims["cp_hidden"]` (1024), but `small_to_mtp_projection` expects `talker_hidden` (2048) as input. For 0.6B both are 1024 so it worked; for 1.7B it's a shape mismatch. **Fix:** Changed to `dims["talker_hidden"]` in both scripts.

3. **ONNX external data scatter:** `torch.onnx.export` creates hundreds of individual tensor files. The original `fix_external_data_ref()` only renamed references without consolidating data, breaking the model. **Fix:** Replaced with `consolidate_external_data()` that loads all scattered data, then saves with `all_tensors_to_one_file=True`.

**Validation results:**
- Vocoder: ONNX vs PyTorch max error = 2.51e-5, mean error = 6.64e-7 ✓ PASS
- LM models: Trace completed cleanly for all 3 (prefill, decode, code predictor)
- Config.json verified: talker hidden_size=2048, cp hidden_size=1024

**Upload:** All artifacts uploaded to `elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX` on HuggingFace.

**Script changes committed:**
- `export_lm.py`: Fixed CP dummy input dimension (talker_hidden, not cp_hidden)
- `reexport_lm_novmap.py`: Fixed CP dummy input + replaced broken `fix_external_data_ref` with `consolidate_external_data`
- `export_vocoder.py`: Added vmap-free masking patch + `--output-dir` flag
- `upload_to_hf.py`: Auto-detect model variant from config.json for README generation

### 2026-04-03: Issue #27 Fix — Remove CP Projection from ONNX Graph

**Root cause:** The `code_predictor.onnx` for 1.7B had `small_to_mtp_projection` (Linear: 2048→1024) baked into the ONNX graph via `CodePredictorWrapper.forward()`. This meant the ONNX model's `inputs_embeds` expected 2048-dim input. For CP prefill, C# feeds the talker hidden state (2048-dim) which got projected down to 1024 — correct. But for subsequent CP decode steps (groups 2-15), C# feeds 1024-dim codec embeddings, which the model tried to project from 2048→1024, causing shape mismatch and truncated output.

**Fix applied to 4 Python export scripts:**
1. `export_lm.py` — Removed `self.projection` from `CodePredictorWrapper` (was leaking ~400 MB unused weights into ONNX). Removed `self.projection(inputs_embeds)` from `forward()`. Changed dummy input dim from `talker_hidden` to `cp_hidden`.
2. `reexport_lm_novmap.py` — Same wrapper fix + dummy dim fix.
3. `reexport_base_novmap.py` — Same wrapper fix (dummy already used `cp_hidden`).
4. `export_embeddings.py` — Added export of `cp_projection_weight.npy` (1024, 2048) and `cp_projection_bias.npy` (1024,) when `small_to_mtp_projection` attribute exists (1.7B only). 0.6B is safely skipped via `hasattr` check.

**Key learning — nn.Module attribute leakage in ONNX:**
Storing an `nn.Module` as `self.projection` in the wrapper causes `torch.onnx.export` to include its parameters in the graph as initializers, even if `forward()` never calls it. This doubled the ONNX data file from 420 MB to 890 MB. Fix: don't store unused submodules as wrapper attributes.

**Re-export results:**
- `code_predictor.onnx`: inputs_embeds shape = [batch, seq, **1024**] (was 2048) ✓
- `code_predictor.onnx.data`: 420 MB (was 428 MB with projection baked in) ✓
- `cp_projection_weight.npy`: (1024, 2048) float32, 8 MB ✓
- `cp_projection_bias.npy`: (1024,) float32, 4 KB ✓

**Uploaded to HuggingFace:** `elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX` — all 4 files.

### 2026-04-03: 1.7B Quality Investigation — Root Causes Identified

**Issue:** 1.7B audio is recognizable but lower quality than expected compared to 0.6B.

**Root cause found — Missing top_k=50 in Code Predictor sampling:**
- `SampleTokenSimple()` (LanguageModel.cs:630) only applies temperature=0.9, no top_k filtering
- Python reference uses `subtalker_top_k=50` via HuggingFace generate (confirmed in generation_config.json and modeling_qwen3_tts.py:2037)
- Without top_k, CP samples from full 2048-token vocab, allowing noisy tail tokens
- CP runs 15 times per talker step; errors compound across groups → audio artifacts

**Secondary issue — Repetition penalty bug:**
- C# always divides by penalty; HuggingFace divides positive logits but multiplies negative logits
- Impact is low with penalty=1.05

**Verified correct:**
- CpProjection: `small_to_mtp_projection` applied uniformly to ALL CP inputs (hidden, group0, codec embeds) — C# matches Python exactly
- Next talker input: CP codec embeddings are 2048-dim for 1.7B (embedding_dim = talker_hidden, not cp_hidden); all 2048 dims accumulated correctly
- codec_head_weight.npy: not needed — codec_head is baked into talker ONNX models
- Sampling params (talker): temperature=0.9, topK=50, topP=1.0, repPenalty=1.05 all match Python defaults

**Key architectural insight confirmed:**
- CP codec embeddings shape = (cp_vocab, talker_hidden) not (cp_vocab, cp_hidden)
- `Qwen3TTSTalkerCodePredictorModel.__init__`: `nn.Embedding(config.vocab_size, embedding_dim)` where `embedding_dim = talker_config.hidden_size`
- This means for 1.7B: CP codec embeddings are (2048, 2048), stored in talker space, projected to CP space (1024) only when fed to the CP model
- For next talker input construction, the raw 2048-dim embeddings are summed WITHOUT projection — this is correct per Python source

### 2026-07-18: Speech Tokenizer Encoder ONNX Export (Issue #32)

**Exported:** `tokenizer12hz_encode.onnx` — the encoder portion of `Qwen3-TTS-Tokenizer-12Hz` for voice cloning ref_text support (ICL mode).

**Architecture:** The encoder is `Qwen3TTSTokenizerV2Encoder`, a subclass of `MimiModel` (from transformers) with decoder parts removed. Pipeline:
1. `MimiEncoder` (SEANet conv encoder) — downsamples audio 960× via strided Conv1d
2. `MimiTransformerModel` (8 layers, sliding_window=250, hidden=512, GeLU) — contextualizes features
3. `MimiConv1d` downsample (2×, converts 25Hz → 12.5Hz frame rate)
4. `MimiSplitResidualVectorQuantizer` (32 RVQ codebooks, keep first 16)

**ONNX tensors:**
- Input: `audio_waveform` — (B, 1, T_samples) float32 — mono 24kHz audio, T must be multiple of 1920
- Output: `audio_codes` — (B, 16, T_frames) int64 — values in [0, 2047], T_frames = T_samples / 1920

**Key technical decisions:**
1. **MimiConv1d padding patched to return 0** — `_get_extra_padding_for_conv1d` uses input-length-dependent math that bakes as constants during tracing. Patched to return 0; C# must pre-pad audio to multiples of 1920.
2. **MimiTransformerModel bypassed** — `create_sliding_window_causal_mask` doesn't trace cleanly. The `EncoderOnnxWrapper` manually builds the sliding-window causal mask from tensor ops, iterates transformer layers directly.
3. **Same compatibility patches as vocoder** — `check_model_inputs`, RoPE `"default"`, `sdpa_mask` scalar fix, `torch.diff` replacement, bool `cumsum` cast, GQA disabled, vmap-free masking.
4. **No external data file** — Model is 209.6 MB (single ONNX file, no .data split needed). The encoder is much smaller than vocoder + LM models.

**Validation results:** ONNX vs PyTorch exact match on all tests:
- Sine wave (5 frames): exact match ✓
- Dynamic axis (2, 10, 50 frames): all exact match ✓
- Real audio file (6.56s Chinese speech, 82 frames): exact match ✓

**MimiModel.encode() shape note:** Returns `(B, num_quantizers, T_frames)` NOT `(num_quantizers, B, T_frames)` — the `_encode_frame` method does `codes.transpose(0, 1)` before returning.

**Files created:**
- `python/export_speech_tokenizer.py` — export script
- `python/onnx_models/tokenizer12hz_encode.onnx` — 209.6 MB ONNX model
- Copied to `%LOCALAPPDATA%\ElBruno\QwenTTS-Base\tokenizer12hz_encode.onnx`

### 2026-04-13: Issue #34 — Missing Compat Patches in export_lm.py

**Root cause identified and fixed:** `export_lm.py` was missing ALL 7 compatibility patches that the other export scripts have. The `RuntimeError: invalid unordered_map<K, T> key` error is the vmap masking crash — `torch.onnx.export` traces through transformers' vmap-based causal mask creation (introduced in transformers 4.57+), which is incompatible with JIT tracing.

**History:** When vmap issues surfaced during 1.7B export, `reexport_lm_novmap.py` was created as a fix, but the original `export_lm.py` was never updated. It only used `attn_implementation="eager"`, which was insufficient with newer transformers.

**Fix applied:**
1. Created `python/compat_patches.py` — shared module with all 7 patches: check_model_inputs compat, ROPE_INIT_FUNCTIONS["default"], sdpa_mask scalar fix, torch.diff ONNX-safe, bool cumsum fix, GQA disable, vmap-free masking registration + helper.
2. Updated `python/export_lm.py` — imports compat_patches, uses vmap-free masking, added model-dir validation with HF repo ID detection.
3. Updated `python/export_embeddings.py` — imports compat_patches, added model-dir validation.
4. Updated `python/README.md` — added Troubleshooting section.

**Key learning — patch duplication debt:** All 4 export scripts had independently copy-pasted the same patches. Creating a shared module prevents this from recurring. The `reexport_*_novmap.py` scripts could also be refactored to use it, but they work as-is.

**The user's "repo ID" issue was a secondary factor:** The scripts expect local model directories, not HuggingFace repo IDs. Added clear error messages to catch this early.
