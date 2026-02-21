---
updated_at: 2026-02-21T17:15:00.000Z
focus_area: C# LanguageModel implementation
active_issues: []
---

# What We're Focused On

**ONNX export complete.** All models exported and validated on NVIDIA A10 GPU machine.

## Current State (10/12 tasks complete)

✅ Model weights downloaded (Qwen3-TTS-12Hz-0.6B-CustomVoice + Tokenizer-12Hz)
✅ Vocoder exported to ONNX (max error 7.97e-06 — PASS)
✅ Talker LM prefill + decode exported to ONNX (stacked KV interface)
✅ Code Predictor exported to ONNX (15 groups, legacy trace)
✅ Embeddings extracted as .npy (text, codec, projections, speaker IDs)
✅ Tokenizer artifacts extracted (vocab, merges, validation cases)
✅ C# TextTokenizer and Vocoder implemented
⚠️ C# LanguageModel.cs — skeleton only
⚠️ C# TtsPipeline.cs — skeleton only

## ONNX Model Summary

| Model | File | Inputs | Outputs | Notes |
|-------|------|--------|---------|-------|
| Vocoder | vocoder.onnx | 1 (codes) | 1 (waveform) | 437 MB |
| Talker Prefill | talker_prefill.onnx | 3 | 58 (logits + hidden + 56 flat KV) | 1.7 GB |
| Talker Decode | talker_decode.onnx | 5 | 4 (logits + hidden + stacked KV) | 1.7 GB |
| Code Predictor | code_predictor.onnx | 4 | 3 (logits + stacked KV) | 420 MB |

## Key Architecture Notes (verified from actual model)

- Talker: 28 layers, 16 attn heads, 8 KV heads, head_dim=128, hidden=1024
- Code Predictor: 5 layers, 16 attn heads, 8 KV heads, head_dim=128, hidden=1024
- num_code_groups=16 (group 0 = Talker, groups 1-15 = Code Predictor)
- Decode KV cache uses stacked format: (num_layers, B, num_kv_heads, T, head_dim)
- Prefill KV cache uses flat format: 56 separate tensors (28 layers × key/value)

## Immediate Next Steps

1. Implement `LanguageModel.cs` — autoregressive inference with KV-cache
2. Wire up `TtsPipeline.cs` end-to-end
3. Test and validate

## Known Issues Resolved

- ✅ `export_lm.py` attribute paths fixed (layers[i].keys/values for DynamicCache)
- ✅ Code Predictor traces cleanly with legacy exporter (15 groups, not 31)
- ⚠️ M-RoPE position ID computation still not implemented in C#
