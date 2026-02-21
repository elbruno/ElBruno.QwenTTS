# Qwen3-TTS Model Architecture Analysis

> Analyzed by Trinity (ML Engineer) from upstream source:
> https://github.com/QwenLM/Qwen3-TTS
>
> Target model: **Qwen3-TTS-0.6B-CustomVoice** + **Qwen3-TTS-Tokenizer-12Hz**

---

## 1. Full Inference Pipeline

```
Text input (str)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  BPE Tokenizer (Qwen3TTSProcessor)                         │
│  "Hello world" → input_ids: [151644, ..., 151645]           │
│  Uses Qwen chat template: <|im_start|>assistant\n{text}...  │
└──────────────────────────┬──────────────────────────────────┘
                           │ input_ids: (1, T_text) int64
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Text Embedding + Projection                                │
│  talker.model.text_embedding → talker.text_projection       │
│  Embeds text tokens, projects from text_hidden_size (2048)  │
│  down to talker hidden_size (1024)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ text_embeds: (1, T_text, 1024) float
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Prefill: Combine text + codec control tokens               │
│  Build input_embeds = text_embeds + codec_prefix_embeds     │
│  Codec prefix: [think/nothink, think_bos, lang_id?,         │
│                 think_eos, speaker_embed?, pad, bos]         │
│  Text is "streamed" — added token-by-token during decode    │
└──────────────────────────┬──────────────────────────────────┘
                           │ combined_embeds: (1, T_prefill, 1024) float
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Talker LM (autoregressive, Qwen3TTSTalkerForConditional)   │
│  20 transformer layers, GQA (16 heads, 2 KV heads)          │
│  M-RoPE position encoding (3D: temporal, height, width)     │
│  hidden_size=1024, intermediate=2048                        │
│  Each step predicts 1st codebook token via codec_head       │
│    → then Code Predictor fills remaining 31 codebooks       │
│  Output: codec_ids (T_audio, 32) int64 per timestep         │
└──────────────────────────┬──────────────────────────────────┘
                           │ talker_codes: (1, T_audio, 32) int64
                           │ values in [0, 2047] from codebook_size=2048
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Vocoder / Speech Tokenizer (Qwen3TTSTokenizerV2Decoder)    │
│  codes → RVQ dequantize → Conv1d → 8-layer Transformer     │
│  → TransConv upsample (2×, 2×) → BigVGAN-style decoder     │
│  → TransConv upsample (8×, 5×, 4×, 3×) → waveform          │
│  Total upsample = 8×5×4×3×2×2 = 1920× (i.e., 12 Hz → 24k) │
└──────────────────────────┬──────────────────────────────────┘
                           │ waveform: (1, num_samples) float32
                           │ sample_rate = 24000 Hz
                           ▼
                    PCM audio output
```

---

## 2. Component Tensor Shapes & Dtypes

### 2a. BPE Tokenizer (Qwen3TTSProcessor)

| I/O | Name | Shape | Dtype | Notes |
|-----|------|-------|-------|-------|
| In  | text | str | — | Chat-formatted text |
| Out | input_ids | `(1, T_text)` | int64 | BPE token IDs |

### 2b. Talker LM (Qwen3TTSTalkerForConditionalGeneration)

**Config (0.6B CustomVoice defaults):**

| Parameter | Value |
|-----------|-------|
| vocab_size | 3072 (codec tokens) |
| hidden_size | 1024 |
| intermediate_size | 2048 |
| num_hidden_layers | 20 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 (GQA, 8:1 ratio) |
| head_dim | 128 (inferred: 16 × 128 = 2048? No — uses explicit head_dim config) |
| text_hidden_size | 2048 |
| num_code_groups | 32 |
| max_position_embeddings | 32768 |
| rope_scaling | M-RoPE (3D interleaved) |

| I/O | Name | Shape | Dtype | Notes |
|-----|------|-------|-------|-------|
| In  | inputs_embeds | `(B, T, 1024)` | float32/bf16 | Combined text + codec embeds |
| In  | attention_mask | `(B, T)` | int64 | 1 = attend, 0 = ignore |
| In  | position_ids | `(3, B, T)` | int64 | M-RoPE: temporal, height, width |
| In  | past_key_values | DynamicCache | float | KV cache (see §4) |
| Out | logits | `(B, 1, 3072)` | float32 | 1st codebook prediction |
| Out | hidden_states | `(B, T, 1024)` | float32/bf16 | Last hidden for code predictor |
| Out | codec_ids | `(B, 1, 32)` | int64 | Full 32-codebook token per step |

### 2c. Code Predictor (Qwen3TTSTalkerCodePredictorModelForConditionalGeneration)

This is a **sub-model** that predicts codebook groups 1–31 given codebook group 0 (which the Talker LM produces).

**Config:**

| Parameter | Value |
|-----------|-------|
| vocab_size | 2048 |
| hidden_size | 1024 |
| intermediate_size | 3072 |
| num_hidden_layers | 5 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| num_code_groups | 32 |

| I/O | Name | Shape | Dtype | Notes |
|-----|------|-------|-------|-------|
| In  | inputs_embeds | `(B, 2, 1024)` | float32/bf16 | [talker_hidden, group_0_embed] at start |
| In  | generation_steps | int | — | Which codebook group to predict (1→31) |
| Out | logits | `(B, 1, 2048)` | float32 | Prediction for next codebook group |
| Out | sequences | `(B, 31)` | int64 | All 31 predicted codebook tokens |

### 2d. Vocoder Decoder (Qwen3TTSTokenizerV2Decoder)

**Config:**

| Parameter | Value |
|-----------|-------|
| num_quantizers | 16 |
| codebook_size | 2048 |
| hidden_size (transformer) | 1024 |
| latent_dim | 1024 |
| num_hidden_layers (transformer) | 8 |
| sliding_window | 72 |
| upsampling_ratios | (2, 2) → 4× pre-upsample |
| upsample_rates | (8, 5, 4, 3) → 480× main upsample |
| decoder_dim | 1536 |
| Total upsample | 4 × 480 = 1920× |

| I/O | Name | Shape | Dtype | Notes |
|-----|------|-------|-------|-------|
| In  | codes | `(B, 16, T_codes)` | int64 | 16 RVQ codebook layers × T timesteps |
| Internal | quantized | `(B, codebook_dim, T_codes)` | float32 | RVQ dequantized |
| Internal | hidden | `(B, T_codes, 1024)` | float32 | After pre_conv + transformer |
| Internal | hidden | `(B, 1024, T_codes×4)` | float32 | After upsampling_ratios (2,2) |
| Internal | wav | `(B, 1536, T_codes×4)` | float32 | Into BigVGAN-style decoder |
| Out | waveform | `(B, 1, T_codes×1920)` | float32 | Clamped to [-1, 1] |

**Note on codebook count:** The Talker LM generates 32 codebook groups per step, but only the first 16 are used by the decoder (per `encoder_valid_num_quantizers=16`). The 32-group structure includes the split RVQ (1 semantic + 15 acoustic quantizers).

### 2e. Speaker Encoder (ECAPA-TDNN, for Base model only — not used in CustomVoice)

| I/O | Name | Shape | Dtype | Notes |
|-----|------|-------|-------|-------|
| In  | mel_spectrogram | `(B, T_mel, 128)` | float32 | 128-dim mel, 24kHz, hop=256 |
| Out | speaker_embedding | `(B, 1024)` | float32 | x-vector style embedding |

---

## 3. How KV-Cache Works

The model uses HuggingFace's `DynamicCache` for both the Talker LM and the Code Predictor.

### Talker LM KV-Cache

1. **Prefill phase:** The full input embedding sequence (text + codec prefix) is processed in one forward pass. Each of the 20 decoder layers stores K and V tensors of shape `(B, num_kv_heads, T_prefill, head_dim)` = `(B, 2, T_prefill, 128)`.

2. **Decode phase:** Each autoregressive step processes a single token. The KV-cache grows by 1 position per step: `(B, 2, T_prefill + step, 128)`.

3. **Position IDs:** Uses M-RoPE (Multimodal Rotary Position Embedding) with 3 axes. For pure text/codec, all three axes share the same value, so it degenerates to standard 1D RoPE. The `rope_deltas` field tracks the position offset due to left-padding.

4. **`past_hidden`**: The Talker also passes `past_hidden = hidden_states[:, -1:, :]` — the last hidden vector — to the Code Predictor at each step, along with the newly generated group-0 codec embedding.

### Code Predictor KV-Cache

1. **Per-step operation:** For each Talker step, the Code Predictor runs its own 31-step autoregressive generation (one for each remaining codebook group).

2. **Prefill:** Takes `[talker_hidden, group_0_embed]` as a 2-token sequence. KV-cache shape per layer: `(B, 8, 2, 128)`.

3. **Generation:** Generates groups 1→31 sequentially, each adding one token to the cache.

4. **Cache is discarded** after each Talker step — the Code Predictor starts fresh for every new timestep.

---

## 4. Multi-Codebook Output Token Structure

The model uses a **32-group multi-codebook** structure with vocabulary size 2048 per group:

```
Step t output: [g0, g1, g2, ..., g31]  — 32 int64 values, each in [0, 2047]
```

### Generation Order

```
Talker LM        →  predicts g0 (via codec_head: Linear(1024 → 3072, top-3072 vocab))
                     Note: talker vocab_size=3072 includes control tokens (EOS, think, pad, bos, etc.)
Code Predictor   →  autoregressively predicts g1, g2, ..., g31
                     Uses 31 separate embedding layers and 31 separate lm_head layers
                     (one per codebook group)
```

### How embeddings combine

At each Talker step, the input embedding is the **sum** of all 32 codebook embeddings from the previous step:

```python
# codec_hiddens = [embed_g0(g0), embed_g1(g1), ..., embed_g31(g31)]
inputs_embeds = codec_hiddens.sum(dim=1, keepdim=True)  # → (B, 1, 1024)
```

Plus a **text hidden state** (trailing text that hasn't been consumed yet):

```python
inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step]
```

### Vocoder input

Only the first 16 codebooks are sent to the vocoder. The codes are transposed to `(B, 16, T)` and passed to the `SplitResidualVectorQuantizer`:
- Codebook 0 → `rvq_first` (1 semantic quantizer)
- Codebooks 1–15 → `rvq_rest` (15 acoustic quantizers)

The dequantized vectors are summed and fed into the decoder pipeline.

---

## 5. Does the Vocoder (Tokenizer-12Hz) Use ONNX Internally?

**No.** The 12Hz tokenizer (`Qwen3TTSTokenizerV2Model`) is implemented entirely in PyTorch. There are no `.onnx` files referenced or loaded in:
- `modeling_qwen3_tts_tokenizer_v2.py`
- `configuration_qwen3_tts_tokenizer_v2.py`

The **25Hz tokenizer** (`tokenizer_v1`) references ONNX in its VQ module (`speech_vq.py`), but the 12Hz tokenizer we target does not.

The 12Hz decoder uses:
- `SplitResidualVectorQuantizer` (pure PyTorch `nn.Embedding` lookups)
- `Qwen3TTSTokenizerV2DecoderTransformerModel` (8-layer transformer, PyTorch)
- BigVGAN-style convolutional decoder with SnakeBeta activations (PyTorch)

No ONNX models are loaded at runtime.

---

## 6. ONNX Export Strategy

### Components to Export

We need **three separate ONNX models** for C# inference:

| # | Component | Source Class | Why Separate |
|---|-----------|-------------|--------------|
| 1 | **Talker LM** | `Qwen3TTSTalkerForConditionalGeneration` | Autoregressive, needs KV-cache, is the largest model |
| 2 | **Code Predictor** | `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` | Called 31× per Talker step, small model, has its own KV-cache |
| 3 | **Vocoder Decoder** | `Qwen3TTSTokenizerV2Decoder` | Non-autoregressive, runs once at the end |

### What We Do NOT Export

- **BPE Tokenizer:** Handled by the `tokenizers` library (or re-implemented in C# using the `vocab.json` + `merges.txt` files). Not a neural network.
- **Speaker Encoder:** Only used by the Base model for voice cloning. CustomVoice uses pre-defined speaker embeddings stored in config.
- **Text Embedding + Projection:** These are embedding lookups + a small MLP. Can be embedded into the Talker LM graph, or done on the C# side.

### Expected Difficulties

1. **KV-Cache Management (Talker LM)**
   - The autoregressive loop with DynamicCache must be split into prefill + decode phases.
   - ONNX doesn't natively support dynamic growing caches. We'll need to export with explicit KV-cache inputs/outputs (standard pattern for LLM export).
   - The 3D M-RoPE position encoding needs careful handling — compute sin/cos tables on the C# side or bake them in.

2. **Code Predictor Loop**
   - The Code Predictor generates 31 tokens autoregressively per Talker step.
   - Each step uses a different `generation_steps` index to select which embedding layer and lm_head to use.
   - Options: (a) export 31 variants, (b) export with `generation_steps` as input and handle routing in ONNX, (c) unroll in C#.
   - **Recommended:** Export one model with `generation_steps` as input. Use conditional logic in C# to feed the right inputs.

3. **Vocoder Causal Convolutions**
   - The `Qwen3TTSTokenizerV2CausalConvNet` uses manual padding logic (`_get_extra_padding_for_conv1d`). This should export cleanly but needs testing.
   - `chunked_decode` logic (300-frame chunks with 25-frame overlap) should be implemented in C#, not baked into ONNX.

4. **Custom Activations**
   - `SnakeBeta` activation: `x + (1/β) * sin²(αx)`. Should export cleanly to ONNX since it's just basic math ops.

5. **Embedding Sum Pattern**
   - The Talker's input is a sum of text embeddings + codec embeddings. This interleaving is orchestrated by the `generate()` method. Must be reimplemented in C#.

### Recommended Approach

**Phase 1: Vocoder First (easiest)**
- Export `Qwen3TTSTokenizerV2Decoder.forward(codes)` where codes shape is `(1, 16, T)`.
- This is a single forward pass, no autoregressive loop.
- Validate output matches PyTorch for sample codes.

**Phase 2: Code Predictor**
- Export with inputs: `(inputs_embeds, generation_steps, past_key_values)`.
- 5 transformer layers, small model. Should export cleanly.
- Validate against PyTorch with same hidden states.

**Phase 3: Talker LM (hardest)**
- Export prefill: `(inputs_embeds, attention_mask, position_ids) → (logits, kv_cache, hidden_states)`.
- Export decode: `(input_ids, attention_mask, position_ids, past_key_values) → (logits, kv_cache, hidden_states)`.
- Validate with sample text → match codec output token-for-token.
- The text embedding lookup + projection + codec embedding lookup + sum should be reimplemented in C# (they're just table lookups + linear layers).

**Phase 4: End-to-End C# Orchestration**
- C# handles: BPE tokenization → build prefill embeddings → Talker LM loop → Code Predictor inner loop → collect codes → Vocoder decode → PCM output.

### Key File Paths (Upstream)

| File | Contains |
|------|----------|
| `qwen_tts/core/models/modeling_qwen3_tts.py` | Talker LM, Code Predictor, Speaker Encoder, ForConditionalGeneration |
| `qwen_tts/core/models/configuration_qwen3_tts.py` | All config classes |
| `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` | Vocoder encoder + decoder |
| `qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py` | Vocoder config |
| `qwen_tts/inference/qwen3_tts_model.py` | High-level inference wrapper |
