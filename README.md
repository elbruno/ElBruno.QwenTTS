# Qwen3-TTS ONNX Pipeline + C# Console App

A project to run **Qwen3-TTS** voice models locally from a C# .NET 10 console application, using ONNX as the bridge between the Python model world and C#.

## Architecture Overview

The pipeline has 3 components:

1. **Text Tokenizer (BPE)** — Converts text + speaker/instruct tags into token IDs
2. **Language Model (Transformer, 0.6B params)** — Autoregressive generation of speech codes (Talker LM + Code Predictor)
3. **Vocoder (ConvNet decoder)** — Converts speech codes → 24kHz audio waveform

**Target model:** [`Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) — 9 built-in speakers, no audio input needed.

Voice cloning with the Base model is a stretch goal.

## Project Structure

```
python/                    # ONNX export pipeline
  export_vocoder.py        # Export vocoder to ONNX
  export_lm.py             # Export Talker LM + Code Predictor to ONNX
  export_embeddings.py     # Extract embedding weights as .npy files
  extract_tokenizer.py     # Extract BPE vocab + merges for C#
  download_models.py       # Download model weights from HuggingFace
  validate_vocoder.py      # Validate ONNX vs PyTorch vocoder output
  ARCHITECTURE.md          # Full architecture reference
  TOKENIZER.md             # Tokenizer + prompt format reference
  requirements.txt         # Python dependencies

src/QwenTTS/               # C# .NET 10 console application
  Program.cs               # CLI entry point (--text, --speaker, --output, --instruct)
  Models/
    TextTokenizer.cs       # BPE tokenizer (Microsoft.ML.Tokenizers)
    LanguageModel.cs       # LM inference with KV-cache (skeleton)
    Vocoder.cs             # ONNX vocoder inference
  Pipeline/
    TtsPipeline.cs         # Full TTS orchestrator (skeleton)
  Audio/
    WavWriter.cs           # WAV file writer (24kHz, 16-bit PCM)

.squad/                    # AI team state (Squad framework)
```

## Current Status

### ✅ Completed (8/12 tasks)

- Python environment with `requirements.txt` + model download script
- Deep architecture analysis (`ARCHITECTURE.md` — 320 lines)
- Tokenizer extraction script + `TOKENIZER.md` documentation
- Vocoder ONNX export script + validation script
- Talker LM + Code Predictor ONNX export scripts (`export_lm.py`, `export_embeddings.py`)
- .NET 10 console app scaffolded with ONNX Runtime 1.24.2, ML.Tokenizers 2.0.0, NAudio 2.2.1
- BPE TextTokenizer fully implemented in C#
- Vocoder ONNX inference wrapper implemented in C#

### ⏳ Pending (4/12 tasks — require GPU machine)

1. **Run Python export scripts** — Download models, generate ONNX files
   - `python download_models.py`
   - `python export_vocoder.py`
   - `python export_lm.py`
   - `python export_embeddings.py`
   - `python extract_tokenizer.py`
2. **Implement LM inference in C#** (`LanguageModel.cs`) — autoregressive loop with KV-cache
3. **Wire up `TtsPipeline.cs`** — full text→tokenize→LM→vocoder→WAV pipeline
4. **End-to-end testing** — validate C# output matches Python reference

## Prerequisites

- .NET 10 SDK
- Python 3.10+ with CUDA support (for export scripts)
- NVIDIA GPU recommended (for model export speed; CPU works but slow)
- ~10GB disk for model weights + ONNX outputs

## How to Continue

```bash
git clone https://github.com/elbruno/qwen-labs-cs.git
cd qwen-labs-cs

# Python setup
cd python
pip install -r requirements.txt
python download_models.py    # Downloads ~4GB of model weights
python export_vocoder.py     # Generate vocoder.onnx
python export_lm.py          # Generate talker_prefill.onnx, talker_decode.onnx, code_predictor.onnx
python export_embeddings.py  # Extract embedding .npy files
python extract_tokenizer.py  # Extract vocab.json + merges.txt

# C# build
cd ../src/QwenTTS
dotnet build
```

## References

- [Qwen3-TTS repo](https://github.com/QwenLM/Qwen3-TTS)
- [HuggingFace model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)

## Squad Team

Morpheus (Lead), Trinity (ML Engineer), Neo (.NET Dev), Tank (Tester)
