# Decision: BPE Tokenizer Extraction Approach

**By:** Trinity (ML Engineer)
**Date:** 2026-02-21
**Status:** Implemented

## What

Created `python/extract_tokenizer.py` to extract the Qwen3-TTS BPE tokenizer artifacts (`vocab.json`, `merges.txt`, configs, and validation cases) for C# consumption. Documented the full prompt format in `python/TOKENIZER.md`.

## Key Findings

1. The tokenizer is a standard Qwen2Tokenizer (GPT-2 BPE) — no exotic preprocessing. C# can re-implement using vocab.json + merges.txt.
2. The 0.6B CustomVoice model does NOT support instruct control (`instruct` is forced to `None`).
3. Speaker selection uses codec embedding IDs (not text tokens): e.g., Ryan=3061 in the Talker's 3072-vocab embedding table.
4. Language is encoded as a codec token inside a think/nothink section; "auto" mode skips the language ID entirely.
5. Eric and Dylan have automatic dialect remapping (Sichuan/Beijing) when language is "chinese" or "auto".

## Impact

Neo can now build the C# BPE tokenizer and prompt construction logic using the documented format and validation cases.
