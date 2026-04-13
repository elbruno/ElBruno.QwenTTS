# Decision: Issue #34 Root Cause — Missing Compat Patches in export_lm.py

**By:** Trinity (ML Engineer)
**Date:** 2026-04-13
**Issue:** #34 — "export lm failed"

## Root Cause

`export_lm.py` was missing ALL the compatibility patches that the other export scripts
(`export_vocoder.py`, `export_speech_tokenizer.py`, `reexport_lm_novmap.py`) have had
since the 1.7B export work.

The error `RuntimeError: invalid unordered_map<K, T> key` is the vmap masking crash —
`torch.onnx.export` traces through transformers' `masking_utils.py` which uses
`torch.vmap` for causal mask creation. This is incompatible with JIT tracing and
crashes in the functorch autograd layer.

## Why It Wasn't Caught

1. The original `export_lm.py` used `attn_implementation="eager"` which bypasses SDPA
   entirely — this worked with the exact transformers version we tested (4.57.3)
2. When vmap issues surfaced during 1.7B export, `reexport_lm_novmap.py` was created
   as a fix, but `export_lm.py` was never updated
3. The user's issue is NOT about repo IDs — it's about the missing patches

## The Fix

1. **Created `python/compat_patches.py`** — centralized all 7 compatibility patches
   (check_model_inputs, ROPE_INIT_FUNCTIONS, sdpa_mask, torch.diff, bool cumsum,
   use_gqa_in_sdpa, vmap-free masking) into one importable module.

2. **Updated `python/export_lm.py`** — imports compat_patches before qwen_tts,
   uses vmap-free masking (sdpa_without_vmap) instead of eager, patches all
   attention layers after model loading. Added model-dir validation with helpful
   error messages for users passing HF repo IDs.

3. **Updated `python/export_embeddings.py`** — imports compat_patches (model loading
   also needs the decorator/ROPE patches). Added model-dir validation.

4. **Updated docs** — added Troubleshooting section to python/README.md explaining
   the vmap error and repo ID requirement.

## Impact

- `export_lm.py` now works with transformers 4.57+ through 5.5+
- Users get clear error messages instead of cryptic stack traces
- The repo ID vs local path confusion is caught early with actionable guidance
- All patches are deduplicated in one shared module for maintainability

## Recommendation

The `reexport_lm_novmap.py` and `reexport_base_novmap.py` scripts can be refactored
to import from `compat_patches.py` too, but they still work as-is. Not urgent.
