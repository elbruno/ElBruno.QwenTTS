# Architecture Overview

## Pipeline

The pipeline has 3 ONNX components:

```
Text --> [BPE Tokenizer] --> token IDs
     --> [Language Model] --> speech codec tokens  (Talker LM + Code Predictor)
     --> [Vocoder]        --> 24kHz WAV audio
```

## Model Components

| Component | Layers | Attn Heads | KV Heads | Hidden | Vocab |
|-----------|--------|-----------|----------|--------|-------|
| Talker LM | 28 | 16 | 8 | 1024 | 3072 |
| Code Predictor | 5 | 16 | 8 | 1024 | 2048 |
| Vocoder | 8 (transformer) | -- | -- | 1024 | -- |

- **Codebook groups**: 16 total (group 0 from Talker, groups 1-15 from Code Predictor)
- **KV cache**: Stacked format `(num_layers, B, num_kv_heads, T, head_dim)` for decode
- **Output**: 24 kHz mono audio (1920x upsample from 12 Hz codec rate)

Source model: [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

## Project Structure

```
setup_environment.py         # One-command setup (download models + restore .NET)
docs/                        # Documentation
  prerequisites.md           # System requirements
  getting-started.md         # Setup and first run
  cli-reference.md           # CLI options and speakers
  architecture.md            # This file
  exporting-models.md        # Re-exporting ONNX models
  troubleshooting.md         # Common issues and fixes

python/                      # ONNX export & download tools
  download_onnx_models.py    # Download pre-exported ONNX models from HF
  upload_to_hf.py            # Upload ONNX models to HuggingFace
  reexport_lm_novmap.py      # Re-export LM models with vmap-free masking
  export_vocoder.py          # Export vocoder to ONNX (advanced)
  export_lm.py               # Export Talker LM + Code Predictor to ONNX (advanced)
  export_embeddings.py       # Extract embedding weights as .npy files (advanced)
  extract_tokenizer.py       # Extract BPE vocab + merges for C# (advanced)
  download_models.py         # Download PyTorch weights from HuggingFace (advanced)
  requirements.txt           # Full deps for ONNX export (torch, transformers, etc.)
  requirements-runtime.txt   # Minimal deps for download only (huggingface_hub)
  ARCHITECTURE.md            # Detailed model architecture reference

src/QwenTTS/                 # C# .NET 10 console application
  Program.cs                 # CLI entry point
  Models/
    TextTokenizer.cs         # BPE tokenizer (Microsoft.ML.Tokenizers)
    LanguageModel.cs         # 3-session autoregressive inference with KV-cache
    Vocoder.cs               # ONNX vocoder inference
    EmbeddingStore.cs        # Text/codec embedding lookups + projection MLP
    NpyReader.cs             # NumPy .npy file loader
  Pipeline/
    TtsPipeline.cs           # Full TTS orchestrator
  Audio/
    WavWriter.cs             # WAV file writer (24 kHz, 16-bit PCM)
```

## Detailed Architecture

For a comprehensive breakdown of tensor shapes, KV-cache management, multi-codebook output structure, and ONNX export strategy, see [`python/ARCHITECTURE.md`](../python/ARCHITECTURE.md).
