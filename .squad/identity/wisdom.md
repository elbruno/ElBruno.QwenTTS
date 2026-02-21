---
last_updated: 2026-02-21T15:28:54.407Z
---

# Team Wisdom

Reusable patterns and heuristics learned through work. NOT transcripts — each entry is a distilled, actionable insight.

## Patterns

<!-- Append entries below. Format: **Pattern:** description. **Context:** when it applies. -->

**Pattern:** ONNX export of autoregressive models requires wrapper nn.Modules that convert DynamicCache to flat tensor I/O.
**Context:** Talker LM and Code Predictor both use HF DynamicCache which can't be traced. Wrappers reconstruct/extract cache inside forward().

**Pattern:** Split autoregressive LM into prefill + decode ONNX models for efficient inference.
**Context:** Prefill handles full sequence (no cache input), decode handles single-token steps (cache in/out). This is standard for LLM ONNX deployment.

**Pattern:** Stack per-group weight matrices into single buffer with index_select for ONNX-friendly dynamic routing.
**Context:** Code Predictor has 31 separate lm_head layers. Stacking into (31, 2048, 1024) and using generation_step index avoids ModuleList tracing issues.

**Pattern:** Use Microsoft.ML.Tokenizers.BpeTokenizer with ByteLevel=true for GPT-2 style tokenizers in C#.
**Context:** Qwen2Tokenizer is GPT-2 BPE. ML.Tokenizers 2.0.0 handles this natively with RegexPreTokenizer.

## Anti-Patterns

<!-- Things we tried that didn't work. **Avoid:** description. **Why:** reason. -->
