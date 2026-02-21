---
date: 2026-02-21
author: neo
status: complete
---

# C# Inference Pipeline Complete

## Decision
Implemented the full C# inference pipeline for Qwen3-TTS with 3 new files (NpyReader, EmbeddingStore, rewritten LanguageModel) and updates to TtsPipeline and Program. The pipeline is now fully wired and ready to test with actual ONNX models.

## Key Implementation Choices

### NpyReader
- Static utility class for loading .npy files (NumPy binary format)
- Supports only the types we need: float32 (`<f4`) and int64 (`<i8`)
- Supports only 1D and 2D arrays (sufficient for embeddings)
- Uses `Buffer.BlockCopy` for efficient data transfer to C# arrays

### EmbeddingStore
- Centralizes all embedding lookups (text, codec, projections)
- Loads config.json for model parameters (codec token IDs, layer counts, etc.)
- Implements text projection as SiLU-gated MLP: `fc2(silu(fc1(x)))`
- Manual matrix-vector multiply (no external BLAS — keep zero runtime deps)

### LanguageModel
- Three ONNX sessions: talker_prefill (initial pass), talker_decode (autoregressive), code_predictor (groups 1-15)
- KV-cache handling: prefill outputs 56 flat tensors (28 layers × key/value), decode uses stacked format (28, B, 8, T, 128)
- Prefill embedding construction follows Python reference exactly: role embed + codec prefix + TTS token embeds + trailing text hidden
- Decode loop: sample group 0 with token suppression ([2048,3071] except codec_eos_token_id), run CP for groups 1-15, sum all embeddings, add trailing text
- Sampling: top-k, temperature, repetition penalty, multinomial with `Random.Shared`

### TtsPipeline
- Removed placeholder BuildPrompt — delegates to TextTokenizer.BuildCustomVoicePrompt()
- Creates EmbeddingStore separately from LanguageModel (LM constructor takes embeddings reference)
- Passes speaker and language to LM.Generate()

### Program
- Added `--model-dir` (required), `--language` (default "auto")
- Wired up full pipeline with try-catch for clean error reporting

## Why This Matters
Bruno can now test end-to-end inference as soon as ONNX models are available. All components are implemented — tokenizer, embeddings, LM, vocoder, WAV writer. The pipeline matches the Python reference architecture exactly.

## Next Steps
1. Test with actual ONNX models from Trinity's export scripts
2. Validate output against Python reference
3. Add GPU execution provider support if needed (currently CPU only)
