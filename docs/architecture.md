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
setup_environment.py             # One-command setup (download models + restore .NET)
QwenTTS.slnx                     # Solution file (all projects)

src/ElBruno.QwenTTS.Core/                # Core library (shared by all apps)
  Pipeline/
    TtsPipeline.cs               # Full TTS orchestrator
  Models/
    TextTokenizer.cs             # BPE tokenizer (Microsoft.ML.Tokenizers)
    LanguageModel.cs             # 3-session autoregressive inference with KV-cache
    Vocoder.cs                   # ONNX vocoder inference
    EmbeddingStore.cs            # Text/codec embedding lookups + projection MLP
    NpyReader.cs                 # NumPy .npy file loader
  Audio/
    WavWriter.cs                 # WAV file writer (24 kHz, 16-bit PCM)

src/ElBruno.QwenTTS/                     # CLI console application
  Program.cs                     # CLI entry point (references QwenTTS.Core)

src/ElBruno.QwenTTS.FileReader/          # Batch file reader (text/SRT → audio)
  Program.cs                     # CLI entry point (references QwenTTS.Core)

src/ElBruno.QwenTTS.Web/                 # Blazor web app — single text-to-speech
  Services/TtsPipelineService.cs # Singleton wrapper with thread-safe access
  Components/Pages/Home.razor    # TTS generation page

src/ElBruno.QwenTTS.Podcast/             # Blazor web app — podcast episode generator
  Services/TtsPipelineService.cs # Singleton wrapper with thread-safe access
  Components/Pages/Podcast.razor # Podcast generation page
  Components/Pages/Samples.razor # LLM prompt templates

docs/                            # Documentation
python/                          # ONNX export & download tools
samples/                         # Sample text files and podcast scripts
```

## Detailed Architecture

For a comprehensive breakdown of tensor shapes, KV-cache management, multi-codebook output structure, and ONNX export strategy, see [`python/ARCHITECTURE.md`](../python/ARCHITECTURE.md).
