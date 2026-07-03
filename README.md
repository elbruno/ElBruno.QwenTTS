# Qwen3-TTS ONNX Pipeline + C# .NET

[![NuGet](https://img.shields.io/nuget/v/ElBruno.QwenTTS.svg?style=flat-square&logo=nuget)](https://www.nuget.org/packages/ElBruno.QwenTTS)
[![NuGet Downloads](https://img.shields.io/nuget/dt/ElBruno.QwenTTS.svg?style=flat-square&logo=nuget)](https://www.nuget.org/packages/ElBruno.QwenTTS)
[![NuGet VoiceCloning](https://img.shields.io/nuget/v/ElBruno.QwenTTS.VoiceCloning.svg?style=flat-square&logo=nuget&label=VoiceCloning)](https://www.nuget.org/packages/ElBruno.QwenTTS.VoiceCloning)
[![Build Status](https://github.com/elbruno/ElBruno.QwenTTS/actions/workflows/publish.yml/badge.svg)](https://github.com/elbruno/ElBruno.QwenTTS/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/elbruno/ElBruno.QwenTTS?style=social)](https://github.com/elbruno/ElBruno.QwenTTS)
[![Twitter Follow](https://img.shields.io/twitter/follow/elbruno?style=social)](https://twitter.com/elbruno)

Run **Qwen3-TTS** text-to-speech locally from C# using ONNX Runtime — no Python needed at inference time. Models are downloaded automatically on first run.

Pre-exported ONNX models are hosted on HuggingFace:
[**elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX**](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX) (0.6B preset voices) |
[**elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX**](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX) (1.7B preset voices + instruct) |
[**elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX**](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX) (voice cloning)

## Features

- **Local TTS Inference** — Run Qwen3-TTS entirely on your machine using ONNX Runtime
- **Multi-Model Support** — Choose between 0.6B (lightweight) and 1.7B (advanced instruct control) variants
- **Automatic Model Download** — Models download from HuggingFace on first run (~5.5 GB for 0.6B, ~10 GB for 1.7B)
- **Instruct Control** — Natural-language style control with 1.7B model (e.g., "speak with excitement", "whisper softly")
- **Multi-Speaker** — 9 built-in voices: ryan, serena, vivian, aiden, eric, dylan, uncle_fu, ono_anna, sohee
- **Voice Cloning** — Clone any voice from a 3-second audio sample ([docs](docs/voice-cloning.md))
- **Web UI** — Blazor app with TTS generation and voice cloning pages ([docs](docs/web-app.md))
- **GPU Acceleration** — Optional CUDA or DirectML support via SessionOptions injection ([docs](docs/gpu-acceleration.md))
- **Multi-Language** — English, Spanish, Chinese, Japanese, Korean, Russian
- **Shared Model Cache** — Models stored once in `%LOCALAPPDATA%/ElBruno/QwenTTS`, shared across all apps
- **Reusable Sessions** — ONNX sessions stay loaded and are reused across requests instead of being recreated
- **Bounded Concurrency + Cancellation** — Configure max concurrent syntheses and cancel queued or in-flight requests at safe boundaries
- **Latency Metrics** — Capture queue, first-audio, and total synthesis latency from the high-level client
- **Streaming Audio Updates** — Emit ordered WAV chunks with format metadata and explicit progressive-vs-chunked capability flags
- **24 kHz WAV Output** — High-quality mono audio

---

## Quick Start

### Install via NuGet

```bash
dotnet add package ElBruno.QwenTTS
```

### Generate speech in C#

```csharp
using ElBruno.QwenTTS.Pipeline;

// 0.6B model (default) — models download automatically (~5.5 GB)
using var pipeline = await TtsPipeline.CreateAsync("models");
await pipeline.SynthesizeAsync("Hello world!", "ryan", "hello.wav", "english");

// 1.7B model — supports instruct control (~10 GB)
using var pipeline17 = await TtsPipeline.CreateAsync("models", variant: QwenModelVariant.Qwen17B);
await pipeline17.SynthesizeAsync("Hello world!", "ryan", "hello.wav", "english",
    instruct: "speak with warmth and excitement");

// Reuse a single pipeline and get per-request latency metrics
using var sharedPipeline = await TtsPipeline.CreateAsync("models", maxConcurrency: 2);
var metrics = await sharedPipeline.SynthesizeWithMetricsAsync(
    "Two callers can queue behind the same shared pipeline.",
    "serena",
    "queued.wav");
Console.WriteLine($"First audio: {metrics.FirstAudioLatency.TotalSeconds:F2}s");

// Stream WAV chunks without building a second full-audio byte[]
await foreach (var update in sharedPipeline.GetStreamingAudioAsync(
    "Stream this response in WAV chunks.",
    "ryan",
    "english"))
{
    if (update.Kind == TextToSpeechUpdateKind.SessionOpen)
    {
        Console.WriteLine($"{update.MediaType} @ {update.SampleRate} Hz, progressive={update.IsProgressive}");
    }

    if (update.Kind == TextToSpeechUpdateKind.AudioChunk)
    {
        await outputStream.WriteAsync(update.AudioData!);
    }
}
```

### CLI

```bash
# Default (0.6B model)
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "Hello, this is a test." --speaker ryan --language english --output hello.wav

# 1.7B model with instruct control
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --variant 1.7b --text "Hello, this is a test." --speaker ryan --instruct "speak with excitement" --output hello.wav
```

Models are downloaded automatically if not present in the `--model-dir` directory.

### Voice Cloning

Clone any voice from a 3-second audio sample using the `ElBruno.QwenTTS.VoiceCloning` package:

```bash
dotnet add package ElBruno.QwenTTS.VoiceCloning
```

```csharp
using ElBruno.QwenTTS.VoiceCloning.Pipeline;

var cloner = await VoiceClonePipeline.CreateAsync();
await cloner.SynthesizeAsync("Hello world!", "reference_speaker.wav", "output.wav", "english");
```

See [docs/voice-cloning.md](docs/voice-cloning.md) for full documentation.

### GPU Acceleration

Pass a `sessionOptionsFactory` to use CUDA or DirectML instead of CPU:

```csharp
using ElBruno.QwenTTS.Pipeline;

// CUDA (NVIDIA) — requires Microsoft.ML.OnnxRuntime.Gpu NuGet package
var tts = await TtsPipeline.CreateAsync(
    sessionOptionsFactory: OrtSessionHelper.CreateCudaOptions);

// DirectML (any GPU on Windows) — requires Microsoft.ML.OnnxRuntime.DirectML NuGet package
// Uses GPU for language model, CPU for vocoder (hybrid mode)
var tts = await TtsPipeline.CreateAsync(
    sessionOptionsFactory: OrtSessionHelper.CreateDirectMlOptions,
    vocoderSessionOptionsFactory: OrtSessionHelper.CreateCpuOptions);
```

See [docs/gpu-acceleration.md](docs/gpu-acceleration.md) for full setup instructions.

## More Examples

```bash
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "Welcome to the future of speech synthesis." --speaker serena --output welcome.wav
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "Speaking with excitement and energy!" --speaker aiden --variant 1.7b --instruct "speak with excitement" --output excited.wav
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "A calm and gentle narration." --speaker ryan --variant 1.7b --instruct "speak slowly and calmly" --output calm.wav
```

### Spanish Examples

```bash
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "Hola, esta es una prueba de texto a voz." --speaker ryan --language spanish --output hola.wav
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "Bienvenidos al futuro de la sintesis de voz." --speaker serena --language spanish --output bienvenidos.wav
```

### Russian Example

```bash
dotnet run --project src/ElBruno.QwenTTS -- --model-dir models --text "Привет, это тест синтеза речи." --speaker ryan --language russian --output russian.wav
```

### File Reader (batch audio from text/SRT files)

```bash
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir models --input samples/hello_demo.txt --speaker ryan --language english --output-dir output/hello
dotnet run --project src/ElBruno.QwenTTS.FileReader -- --model-dir models --input samples/demo_subtitles.srt --speaker serena --output-dir output/subtitles
```

### Web App (browser UI)

```bash
dotnet run --project src/ElBruno.QwenTTS.Web
```

Open [http://localhost:5153](http://localhost:5153) — two pages:
- **🔊 TTS** — type text or upload files, pick a voice, and generate speech
- **🎭 Voice Clone** — record your voice or upload a WAV, then synthesize with your cloned voice

---

## Documentation

| Document | Description |
|----------|-------------|
| [Prerequisites](docs/prerequisites.md) | System requirements (.NET 8+/10, disk space) |
| [Getting Started](docs/getting-started.md) | Setup, auto-download, and first run |
| [Core Library](docs/core-library.md) | ElBruno.QwenTTS API reference and usage examples |
| [CLI Reference](docs/cli-reference.md) | All command options, speakers, and examples |
| [File Reader](docs/file-reader.md) | Batch audio generation from text and SRT files |
| [Web App](docs/web-app.md) | Blazor web UI for speech generation |
| [Architecture](docs/architecture.md) | Pipeline design, model components, project structure |
| [Exporting Models](docs/exporting-models.md) | Re-exporting ONNX models from PyTorch weights |
| [Voice Cloning](docs/voice-cloning.md) | Clone any voice from a 3-second reference audio |
| [GPU Acceleration](docs/gpu-acceleration.md) | CUDA, DirectML, and CPU configuration |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [Detailed Architecture](python/ARCHITECTURE.md) | Full tensor shapes, KV-cache, codebook structure |
| [Changelog](CHANGELOG.md) | Versioned summary of notable changes |

## Python Tools

The `python/` directory contains tools for **exporting ONNX models from PyTorch weights** and **downloading models from HuggingFace**. These are only needed if you want to re-export or customize models — they are not required for running the C# pipeline.

---

## Building from Source

```bash
git clone https://github.com/elbruno/ElBruno.QwenTTS.git
cd ElBruno.QwenTTS
dotnet build
dotnet test
```

## Requirements

- .NET 8.0 or .NET 10.0 SDK
- ONNX Runtime compatible platform (Windows, Linux, macOS)
- ~5.5 GB disk space for model files

---

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a branch** for your feature or fix: `git checkout -b feature/my-feature`
3. **Make your changes** and ensure the solution builds: `dotnet build`
4. **Run tests**: `dotnet test`
5. **Submit a pull request** with a clear description of the changes

Please open an issue first for major changes or new features to discuss the approach.

---

## Related Projects

- [**ElBruno.PersonaPlex**](https://github.com/elbruno/ElBruno.PersonaPlex) — NVIDIA PersonaPlex-7B full-duplex speech-to-speech for local C# inference via ONNX Runtime. Pre-exported ONNX models: [elbruno/personaplex-7b-v1-onnx](https://huggingface.co/elbruno/personaplex-7b-v1-onnx)

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Original model (PyTorch)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [Pre-exported ONNX models](https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX)

---

## 👋 About the Author

Hi! I'm **ElBruno** 🧡, a passionate developer and content creator exploring AI, .NET, and modern development practices.

**Made with ❤️ by [ElBruno](https://github.com/elbruno)**

If you like this project, consider following my work across platforms:

- 📻 **Podcast**: [No Tienen Nombre](https://notienenombre.com) — Spanish-language episodes on AI, development, and tech culture
- 💻 **Blog**: [ElBruno.com](https://elbruno.com) — Deep dives on embeddings, RAG, .NET, and local AI
- 📺 **YouTube**: [youtube.com/elbruno](https://www.youtube.com/elbruno) — Demos, tutorials, and live coding
- 🔗 **LinkedIn**: [@elbruno](https://www.linkedin.com/in/elbruno/) — Professional updates and insights
- 𝕏 **Twitter**: [@elbruno](https://www.x.com/in/elbruno/) — Quick tips, releases, and tech news

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
