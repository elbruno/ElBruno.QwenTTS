# Decision: Speech Tokenizer Encoder ONNX Export Strategy

**Date:** 2026-07-18
**Author:** Trinity (ML Engineer)
**Status:** Implemented
**Issue:** #32 (ref_text voice cloning support)

## Context

Voice cloning ICL (In-Context Learning) mode requires encoding reference audio into codebook tokens. The speech tokenizer encoder (Qwen3-TTS-Tokenizer-12Hz) was only available in PyTorch. We needed an ONNX export for the C# runtime.

## Decision

Export the encoder portion of `Qwen3TTSTokenizerV2Encoder` (which extends `MimiModel` from transformers) to ONNX as `tokenizer12hz_encode.onnx`.

### Key design choices:

1. **Wrapper approach** — Created `EncoderOnnxWrapper` that bypasses `create_sliding_window_causal_mask` and manually builds the attention mask. Same pattern as the vocoder export's `VocoderOnnxWrapper`.

2. **Input constraint** — Audio length must be a multiple of 1920 samples (one frame at 12.5 Hz). This simplifies the ONNX graph by eliminating data-dependent padding. The C# runtime is responsible for padding.

3. **Single file, no split** — At 209.6 MB the model fits in a single ONNX file (no external .data needed).

4. **16 of 32 codebooks** — The RVQ has 32 codebooks but only the first 16 are valid for the 12Hz tokenizer. The wrapper enforces this at export time.

## Impact on C# Side

- **New file needed:** `tokenizer12hz_encode.onnx` must be downloaded alongside existing model files
- **Input prep:** C# must pad audio to multiples of 1920 samples before inference
- **Output:** (B, 16, T_frames) int64 codes — ready to feed into the TTS pipeline for ICL mode
