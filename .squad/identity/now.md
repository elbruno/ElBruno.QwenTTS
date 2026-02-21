---
updated_at: 2026-02-21T16:30:00.000Z
focus_area: GPU handoff — ONNX export + C# LM implementation
active_issues: []
---

# What We're Focused On

**Handoff in progress.** Moving from non-GPU machine to NVIDIA GPU machine.

## Current State (8/12 tasks complete)

All Python export SCRIPTS are written but NOT yet run (need model weights downloaded on GPU machine).
All C# scaffolding is in place — TextTokenizer and Vocoder are fully implemented. LanguageModel and TtsPipeline are skeletons.

## Immediate Next Steps (on GPU machine)

1. Run `python/download_models.py` to get model weights
2. Run all export scripts (vocoder → LM → embeddings → tokenizer)
3. Validate ONNX outputs match PyTorch
4. Implement `LanguageModel.cs` — autoregressive inference with KV-cache (40 tensors for Talker, 10 for Code Predictor)
5. Wire up `TtsPipeline.cs` end-to-end
6. Test and validate

## Key Architecture Docs

- `python/ARCHITECTURE.md` — Full model internals (tensor shapes, KV-cache, M-RoPE)
- `python/TOKENIZER.md` — BPE format, chat template, special tokens, prompt construction

## Known Risks

- `export_lm.py` may need attribute path adjustments when running against actual model weights
- Code Predictor per-group embedding routing may not trace cleanly to ONNX
- M-RoPE position ID computation not yet implemented in C#
