# Qwen3-TTS ONNX Pipeline + C# Console App

Run **Qwen3-TTS** text-to-speech locally from a C# .NET 10 console application using ONNX Runtime — no Python needed at inference time.

Pre-exported ONNX models are hosted on HuggingFace:
[**elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX**](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX)

---

## Quick Start (one command)

```bash
# Clone the repo, then run:
python setup_environment.py
```

This single script handles everything:

1. ✅ Checks prerequisites (Python ≥ 3.10, .NET 10 SDK)
2. ✅ Installs `huggingface_hub` (if missing)
3. ✅ Downloads ONNX models from HuggingFace (~5.5 GB)
4. ✅ Verifies all model files are present
5. ✅ Restores .NET NuGet packages

Then generate speech:

```bash
dotnet run --project src/QwenTTS -- \
  --model-dir python/onnx_runtime \
  --text "Hello, this is a test." \
  --speaker ryan \
  --language english \
  --output hello.wav
```

---

## Prerequisites

| Requirement | Version | Install |
|-------------|---------|---------|
| **Python** | ≥ 3.10 | [python.org](https://www.python.org/downloads/) |
| **.NET SDK** | 10.0 | [dotnet.microsoft.com](https://dotnet.microsoft.com/download) |
| **Disk space** | ~6 GB | For ONNX models + NuGet packages |

> **No GPU required** — inference runs on CPU via ONNX Runtime. GPU acceleration available by switching to `Microsoft.ML.OnnxRuntime.Gpu`.

---

## Step-by-Step Setup (manual)

If you prefer manual steps over the setup script:

### 1. Download ONNX models

```bash
cd python
pip install huggingface_hub
python download_onnx_models.py
```

Options:
```bash
# Custom output directory
python download_onnx_models.py --output-dir ./my-models

# Different HuggingFace repo
python download_onnx_models.py --repo-id your-user/your-repo
```

This creates the following directory structure:

```
python/onnx_runtime/
├── talker_prefill.onnx + .data    (~1.7 GB — Talker LM prefill)
├── talker_decode.onnx + .data     (~1.7 GB — Talker LM decode)
├── code_predictor.onnx            (~420 MB — Code Predictor)
├── vocoder.onnx + .data           (~437 MB — Vocoder decoder)
├── embeddings/                    (text/codec embeddings, config, speaker IDs)
│   ├── config.json
│   ├── text_embedding.npy
│   ├── talker_codec_embedding.npy
│   ├── cp_codec_embedding_0..14.npy
│   ├── text_projection_fc*.npy
│   ├── codec_head_weight.npy
│   └── speaker_ids.json
└── tokenizer/
    ├── vocab.json
    └── merges.txt
```

### 2. Build the C# app

```bash
dotnet restore src/QwenTTS/QwenTTS.csproj
dotnet build src/QwenTTS/QwenTTS.csproj
```

### 3. Generate speech

```bash
dotnet run --project src/QwenTTS -- \
  --model-dir python/onnx_runtime \
  --text "Hello world" \
  --speaker ryan \
  --language english \
  --output hello.wav
```

---

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-dir` | *(required)* | Path to downloaded model directory |
| `--text` | *(required)* | Text to synthesize |
| `--speaker` | `Ryan` | Speaker voice name (see below) |
| `--language` | `auto` | Language: `english`, `chinese`, `auto`, etc. |
| `--output` | `output.wav` | Output WAV file path |
| `--instruct` | *(none)* | Voice style instruction (e.g., "speak happily") |

### Available Speakers

| Speaker | Notes |
|---------|-------|
| `ryan` | English male |
| `serena` | English female |
| `vivian` | English female |
| `aiden` | English male |
| `eric` | Sichuan dialect |
| `dylan` | Beijing dialect |
| `uncle_fu` | Chinese male |
| `ono_anna` | Japanese female |
| `sohee` | Korean female |

---

## Architecture Overview

The pipeline has 3 ONNX components:

```
Text → [BPE Tokenizer] → token IDs
     → [Language Model] → speech codec tokens  (Talker LM + Code Predictor)
     → [Vocoder]        → 24kHz WAV audio
```

| Component | Layers | Attn Heads | KV Heads | Hidden | Vocab |
|-----------|--------|-----------|----------|--------|-------|
| Talker LM | 28 | 16 | 8 | 1024 | 3072 |
| Code Predictor | 5 | 16 | 8 | 1024 | 2048 |
| Vocoder | 8 (transformer) | — | — | 1024 | — |

- **Codebook groups**: 16 total (group 0 from Talker, groups 1–15 from Code Predictor)
- **KV cache**: Stacked format `(num_layers, B, num_kv_heads, T, head_dim)` for decode
- **Output**: 24 kHz mono audio (1920× upsample from 12 Hz codec rate)

Source model: [`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

---

## Project Structure

```
setup_environment.py         # ⭐ One-command setup (download models + restore .NET)

python/                      # ONNX export & download tools
  download_onnx_models.py    # Download pre-exported ONNX models from HF
  upload_to_hf.py            # Upload ONNX models to HuggingFace
  export_vocoder.py          # Export vocoder to ONNX (advanced)
  export_lm.py               # Export Talker LM + Code Predictor to ONNX (advanced)
  export_embeddings.py       # Extract embedding weights as .npy files (advanced)
  extract_tokenizer.py       # Extract BPE vocab + merges for C# (advanced)
  download_models.py         # Download PyTorch weights from HuggingFace (advanced)
  requirements.txt           # Full deps for ONNX export (torch, transformers, etc.)
  requirements-runtime.txt   # Minimal deps for download only (huggingface_hub)
  ARCHITECTURE.md            # Full architecture reference

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

---

## Re-exporting Models (Advanced)

If you need to re-export the ONNX models from PyTorch weights (e.g., for a different model variant):

```bash
cd python
pip install -r requirements.txt       # Installs torch, transformers, onnx, etc.
python download_models.py              # Download PyTorch weights (~4 GB)
python reexport_lm_novmap.py           # → onnx_runtime/ (prefill, decode, code_predictor)
python export_vocoder.py               # → onnx_models/vocoder.onnx
python export_embeddings.py            # → onnx_models/embeddings/
python extract_tokenizer.py            # → tokenizer_artifacts/
```

> **Note:** The LM models must be exported with vmap-free masking (`reexport_lm_novmap.py`)
> to work with C# ONNX Runtime. The standard `export_lm.py` produces models that use
> `torch.vmap`-generated Expand nodes which are incompatible with ORT's C# binding.

To upload your exported models to HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli login                  # Needs a write token from https://huggingface.co/settings/tokens
python upload_to_hf.py --repo-id your-username/your-repo-name
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python: command not found` | Install Python 3.10+ and add to PATH |
| `dotnet: command not found` | Install [.NET 10 SDK](https://dotnet.microsoft.com/download) |
| Download hangs or fails | Re-run `python download_onnx_models.py` — it skips already-downloaded files |
| `huggingface_hub` import error | Run `pip install huggingface_hub` |
| Out of memory during inference | Close other applications — the LM loads ~3.4 GB into RAM |
| Wrong speaker name | Use lowercase: `ryan`, `serena`, etc. (see speaker table above) |
| `invalid expand shape` in C# ORT | Re-export LM models with `python/reexport_lm_novmap.py` (vmap-free masking) |
| `Not a valid NPY file (bad magic)` | Ensure NpyReader uses byte array `[0x93, ...]` not UTF-8 string literal |

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Original model (PyTorch)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [Pre-exported ONNX models](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX)

## Squad Team

Morpheus (Lead), Trinity (ML Engineer), Neo (.NET Dev), Tank (Tester)
