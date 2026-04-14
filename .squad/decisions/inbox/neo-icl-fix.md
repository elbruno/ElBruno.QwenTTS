# Decision: ICL Embedding Ordering in BuildPrefillEmbedding

**Date:** 2026-07-15  
**Author:** Neo (.NET Developer)  
**Issue:** #36  
**Status:** Accepted  

## Context

The ICL (In-Context Learning) voice cloning path in `BuildPrefillEmbedding` (`LanguageModel.cs`) had three bugs that caused incorrect embeddings compared to the official Qwen3-TTS `generate_icl_prompt` implementation.

## Decision

Align the C# ICL embedding construction with the official Qwen3-TTS Python implementation:

1. **Token ordering:** The "last position" (tts_bos + codec_prefix[-2]) is part of the codec prefix and must come BEFORE the ICL section, not after it. The ICL section then appends: ref_text + target_text + tts_eos (each paired with codec_pad), followed by codec_bos + ref_audio_codes (each paired with tts_pad).

2. **Codec embeddings for ref audio:** Group 0 uses `TalkerCodecEmbedding` (same embedding space as the talker model). Groups 1-15 use `CpCodecEmbedding(g-1, ...)` (Code Predictor embedding space, potentially different dimension). Accumulate only `_cpHiddenSize` elements for CP groups.

3. **Trailing text hidden:** In ICL mode, all text (ref + target + eos) is included in the prefill, so trailing text hidden is just `[ttsPadProj]`. The non-ICL path continues to use `tokens[4:-5] + tts_eos`.

4. **Non-ICL guard:** The `first_text + codec_bos` append step and standard trailing text only apply when ICL data is absent.

## Consequences

- ICL voice cloning will now produce correct embeddings matching the reference implementation.
- The `_cpHiddenSize` vs `_hiddenSize` distinction is critical for 1.7B models where these differ (1024 vs 2048).
- Non-ICL (standard speaker) path is unchanged — full backward compatibility.
