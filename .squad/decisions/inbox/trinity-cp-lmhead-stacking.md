# Decision: Code Predictor lm_head stacking for ONNX

**By:** Trinity (ML Engineer)
**Date:** 2026-02-21
**Status:** Implemented

## Context

The Code Predictor has 31 separate `nn.Linear` lm_head modules (one per codebook group 1-31). ONNX tracing cannot dynamically index into a `nn.ModuleList`.

## Decision

Stack all 31 lm_head weight matrices into a single `(31, 2048, 1024)` buffer and use `torch.index_select` on the `generation_steps` input to select the correct weight at runtime. The matmul `hidden_states @ weight.T` then produces the logits.

## Alternatives considered

1. **Export 31 separate models** — rejected; too many files, complicates C# orchestration.
2. **torch.where with all 31 outputs** — rejected; wasteful computation (runs all heads, discards 30).
3. **Script-mode export with if/else** — rejected; `torch.jit.script` has poor support for HF model internals.

## Impact

- C# calls one `code_predictor.onnx` model per step, passing `generation_steps` (0-30) to select the head.
- The same approach works for the 31 codec embeddings, but those are extracted as raw .npy files for C# lookup.
