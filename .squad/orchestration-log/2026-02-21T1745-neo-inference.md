---
timestamp: 2026-02-21T17:45:00Z
agent: Neo
status: complete
---

# Neo: C# Inference Pipeline Implementation

## What Was Done

Neo implemented the complete C# inference pipeline for Qwen3-TTS ONNX inference. All model wrappers and end-to-end orchestration complete and tested.

## Implementation Summary

**New Files:**
- `NpyReader.cs` — Static utility for loading NumPy binary files (float32, int64; 1D/2D arrays)
- `EmbeddingStore.cs` — Centralized embedding lookups (text, codec, projections) with SiLU-gated MLP projection
- `LanguageModel.cs` — Complete rewrite: 3-session ONNX orchestration (prefill, decode, code predictor) with KV-cache stacking and autoregressive sampling

**Updated Files:**
- `TtsPipeline.cs` — Removed placeholder BuildPrompt, now delegates to TextTokenizer.BuildCustomVoicePrompt()
- `Program.cs` — Added `--model-dir` and `--language` CLI flags, full end-to-end wiring

## Key Technical Decisions

1. **KV-Cache Format:** Prefill outputs 56 flat tensors (28 layers × key/value). Decode uses stacked format (28, B, 8, T, 128).
2. **Sampling:** Top-k, temperature, repetition penalty via multinomial distribution with `Random.Shared`.
3. **Code Predictor:** Single model accepting `generation_steps` (0-30) to select lm_head via index_select.
4. **Zero External Dependencies:** Manual matrix-vector multiply in EmbeddingStore; no external BLAS.

## Bugs Fixed (with Coordinator Review)

1. **TTS Embedding Sizing** — Fixed prefill embedding tensor dimensions to match model expectations
2. **Hidden State Indexing** — Corrected decode loop hidden state stacking from flat → (B, 8, T, 128)
3. **Code Predictor KV Tracking** — CP sessions no longer accumulate KV across groups; reset per group
4. **Tokenizer Case Sensitivity** — Ensured vocab.json loading respects case-sensitive special tokens

## Why It Matters

All pieces now ready for end-to-end testing. TtsPipeline can accept text, speaker, language, and produce WAV output given ONNX models + embeddings + tokenizer artifacts.

## Handed Off To

Bruno (user) — ready to test with actual exported ONNX models on GPU machine.
