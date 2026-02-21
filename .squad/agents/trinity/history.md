# Project Context

- **Owner:** Bruno Capuano
- **Project:** Qwen3-TTS → ONNX → C# .NET 10 console app for local voice generation
- **Stack:** Python (PyTorch, transformers, ONNX), C# (.NET 10, ONNX Runtime), Qwen3-TTS
- **Created:** 2026-02-21T15:38Z

## Learnings

<!-- Append new learnings below. Each entry is something lasting about the project. -->

### 2026-02-21: Qwen3-TTS Architecture Deep Dive

**Pipeline:** Text → BPE → Text Embedding (2048d) → Projection (→1024d) → Talker LM (20-layer, GQA 16/2, M-RoPE) → Code Predictor (5-layer, 31 autogressive steps per Talker step) → 32-codebook codes → Vocoder (16 codebooks used, RVQ dequantize → 8-layer transformer → BigVGAN-style conv decoder, 1920× upsample → 24kHz PCM)

**Key architecture facts:**
- Talker LM: hidden_size=1024, vocab_size=3072, num_code_groups=32
- Code Predictor: 5 layers, hidden_size=1024, vocab_size=2048, predicts groups 1-31 sequentially
- Vocoder (Tokenizer-12Hz): Decoder only (no ONNX internally), 16 RVQ codebooks, codebook_size=2048, sliding_window=72
- 12Hz tokenizer is pure PyTorch — no ONNX files loaded at runtime
- 25Hz tokenizer (not our target) DOES use ONNX internally
- Total upsample factor = 1920× (12 Hz codes → 24kHz audio)

**ONNX export plan:** 3 models — Vocoder (easiest, single pass), Code Predictor (small, 31-step loop), Talker LM (largest, KV-cache).

**Key source files:**
- `qwen_tts/core/models/modeling_qwen3_tts.py` — Talker LM + Code Predictor + Speaker Encoder
- `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` — Vocoder
- `qwen_tts/inference/qwen3_tts_model.py` — Inference orchestration

**Environment:** `python/` directory created with requirements.txt, download_models.py, README.md, ARCHITECTURE.md
