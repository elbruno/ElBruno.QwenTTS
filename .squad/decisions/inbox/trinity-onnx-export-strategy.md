### ONNX Export Strategy: 3-Model Split

**By:** Trinity (ML Engineer)
**Date:** 2026-02-21
**Status:** Proposed

**What:** Export Qwen3-TTS as three separate ONNX models: Vocoder (single-pass), Code Predictor (small autoregressive), and Talker LM (large autoregressive with KV-cache). Export order: Vocoder → Code Predictor → Talker LM.

**Why:**
- The model has three distinct components with different execution patterns (single-pass vs autoregressive)
- The Code Predictor runs 31× per Talker step — must be fast and separate
- KV-cache handling differs between Talker (grows across the whole generation) and Code Predictor (reset each step)
- The vocoder is the simplest to validate and provides a good first milestone

**Impact:**
- C# side must orchestrate the autoregressive loop, embedding lookups, and codec assembly
- BPE tokenization will be handled in C# (not an ONNX model)
- Text embeddings + projection can be embedded in Talker ONNX graph or done in C#

**12Hz tokenizer is pure PyTorch** — no pre-existing ONNX files to reuse. Everything must be exported fresh.
