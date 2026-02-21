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
