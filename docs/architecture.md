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
setup_environment.py             # Alternative Python setup (optional)
ElBruno.QwenTTS.slnx             # Solution file (all projects)

src/ElBruno.QwenTTS.Core/                # Core library (shared by all apps)
  Pipeline/
    TtsPipeline.cs               # Full TTS orchestrator + CreateAsync factory
    ModelDownloader.cs           # Auto-download models from HuggingFace
  Models/
    TextTokenizer.cs             # BPE tokenizer (Microsoft.ML.Tokenizers)
    LanguageModel.cs             # 3-session autoregressive inference with KV-cache
    Vocoder.cs                   # ONNX vocoder inference
    EmbeddingStore.cs            # Text/codec embedding lookups + projection MLP
    NpyReader.cs                 # NumPy .npy file loader
  Audio/
    WavWriter.cs                 # WAV file writer (24 kHz, 16-bit PCM)

src/ElBruno.QwenTTS.Core.Tests/          # Unit tests (xUnit)

src/ElBruno.QwenTTS/                     # CLI console application

src/ElBruno.QwenTTS.FileReader/          # Batch file reader (text/SRT → audio)

src/ElBruno.QwenTTS.Web/                 # Blazor web app — single text-to-speech

docs/                            # Documentation
python/                          # ONNX export & download tools (optional)
samples/                         # Sample text files and podcast scripts
```

## Detailed Architecture

For a comprehensive breakdown of tensor shapes, KV-cache management, multi-codebook output structure, and ONNX export strategy, see [`python/ARCHITECTURE.md`](../python/ARCHITECTURE.md).
