# Qwen3-TTS ONNX Pipeline + C# Console App

A project to run **Qwen3-TTS** voice models locally from a C# .NET 10 console application, using ONNX as the bridge between the Python model world and C#.

## Architecture Overview

The pipeline has 3 components:

1. **Text Tokenizer (BPE)** — Converts text + speaker/instruct tags into token IDs
2. **Language Model (Transformer, 0.6B params)** — Autoregressive generation of speech codes (Talker LM + Code Predictor)
3. **Vocoder (ConvNet decoder)** — Converts speech codes → 24kHz audio waveform

**Target model:** [`Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) — 9 built-in speakers, no audio input needed.

## Quick Start

### 1. Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- Python 3.10+ with `huggingface_hub` (`pip install huggingface_hub`)

### 2. Download ONNX Models

The ONNX models (~5.5 GB) are hosted on HuggingFace. Download them with:

```bash
cd python
pip install huggingface_hub
python download_onnx_models.py
```

This downloads to `python/onnx_runtime/` with the required directory structure:

```
python/onnx_runtime/
  talker_prefill.onnx + .data    (Talker LM prefill, ~1.7 GB)
  talker_decode.onnx + .data     (Talker LM decode, ~1.7 GB)
  code_predictor.onnx            (Code Predictor, ~420 MB)
  vocoder.onnx + .data           (Vocoder decoder, ~437 MB)
  embeddings/                    (text/codec embeddings, config, speaker IDs)
  tokenizer/                     (vocab.json, merges.txt)
```

### 3. Build and Run

```bash
cd src/QwenTTS
dotnet build

# Generate speech
dotnet run -- --model-dir ../../python/onnx_runtime --text "Hello world" --speaker ryan --language english --output hello.wav
```

### Available Speakers

`serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | (required) | Path to model directory |
| `--text` | (required) | Text to synthesize |
| `--speaker` | `Ryan` | Speaker name |
| `--language` | `auto` | Language (`english`, `chinese`, `auto`, etc.) |
| `--output` | `output.wav` | Output WAV file path |
| `--instruct` | (none) | Voice style instruction (e.g., "speak happily") |

## Project Structure

```
python/                    # ONNX export pipeline
  download_onnx_models.py  # ⭐ Download pre-exported ONNX models from HF
  upload_to_hf.py          # Upload ONNX models to HuggingFace
  export_vocoder.py        # Export vocoder to ONNX
  export_lm.py             # Export Talker LM + Code Predictor to ONNX
  export_embeddings.py     # Extract embedding weights as .npy files
  extract_tokenizer.py     # Extract BPE vocab + merges for C#
  download_models.py       # Download PyTorch weights from HuggingFace
  ARCHITECTURE.md          # Full architecture reference

src/QwenTTS/               # C# .NET 10 console application
  Program.cs               # CLI entry point
  Models/
    TextTokenizer.cs       # BPE tokenizer (Microsoft.ML.Tokenizers)
    LanguageModel.cs       # 3-session autoregressive inference with KV-cache
    Vocoder.cs             # ONNX vocoder inference
    EmbeddingStore.cs      # Text/codec embedding lookups + projection MLP
    NpyReader.cs           # NumPy .npy file loader
  Pipeline/
    TtsPipeline.cs         # Full TTS orchestrator
  Audio/
    WavWriter.cs           # WAV file writer (24kHz, 16-bit PCM)

.squad/                    # AI team state (Squad framework)
```

## Re-exporting Models (Advanced)

If you need to re-export the ONNX models from PyTorch weights (e.g., for a different model variant):

```bash
cd python
pip install -r requirements.txt
python download_models.py       # Download PyTorch weights (~4 GB)
python export_vocoder.py        # → onnx_models/vocoder.onnx
python export_lm.py             # → onnx_models/talker_prefill.onnx, talker_decode.onnx, code_predictor.onnx
python export_embeddings.py     # → onnx_models/embeddings/
python extract_tokenizer.py     # → tokenizer_artifacts/

# Upload to HuggingFace (needs write token: huggingface-cli login)
python upload_to_hf.py --repo-id your-username/your-repo-name
```

## Model Architecture (verified)

| Component | Layers | Attn Heads | KV Heads | Hidden | Vocab |
|-----------|--------|-----------|----------|--------|-------|
| Talker LM | 28 | 16 | 8 | 1024 | 3072 |
| Code Predictor | 5 | 16 | 8 | 1024 | 2048 |
| Vocoder | 8 (transformer) | — | — | 1024 | — |

- **Codebook groups**: 16 total (group 0 from Talker, groups 1-15 from Code Predictor)
- **KV cache**: Stacked format `(num_layers, B, num_kv_heads, T, head_dim)` for decode
- **Output**: 24kHz mono audio (1920× upsample from 12Hz codec rate)

## References

- [Qwen3-TTS repo](https://github.com/QwenLM/Qwen3-TTS)
- [HuggingFace model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

## Squad Team

Morpheus (Lead), Trinity (ML Engineer), Neo (.NET Dev), Tank (Tester)
