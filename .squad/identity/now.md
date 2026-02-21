---
updated_at: 2026-02-21T17:45:00.000Z
focus_area: End-to-end testing
active_issues: []
---

# What We're Focused On

**C# inference pipeline implemented.** All models exported to ONNX and full C# pipeline built.

## Current State (12/12 tasks complete — ready for E2E testing)

✅ Model weights downloaded (Qwen3-TTS-12Hz-0.6B-CustomVoice + Tokenizer-12Hz)
✅ Vocoder exported to ONNX (max error 7.97e-06 — PASS)
✅ Talker LM prefill + decode exported to ONNX (stacked KV interface)
✅ Code Predictor exported to ONNX (15 groups, legacy trace)
✅ Embeddings extracted as .npy (text, codec, projections, speaker IDs)
✅ Tokenizer artifacts extracted (vocab, merges, validation cases)
✅ C# TextTokenizer and Vocoder implemented
✅ C# NpyReader for loading .npy files
✅ C# EmbeddingStore with text/codec lookups and projection MLP
✅ C# LanguageModel with 3-session autoregressive inference
✅ C# TtsPipeline wired end-to-end
✅ C# Program.cs with --model-dir CLI

## Next Step

Run end-to-end: copy ONNX models + embeddings + tokenizer into model directory, run the app.
